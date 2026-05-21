// anywidget ESM for spatialdata_plot.pl.interactive.DrawCanvas.
// Pure client-side drawing on an SVG overlay above the rendered image PNG.
// Shape geometry (in image-pixel coordinates) is synced back to Python via
// the `shapes` traitlet; conversion to data/CS coords happens server-side.

function render({ model, el }) {
    const W = model.get("image_width");
    const H = model.get("image_height");
    const maxW = model.get("max_display_width") || 880;

    const wrap = document.createElement("div");
    wrap.style.cssText = `
        display: block;
        width: 100%;
        max-width: ${maxW}px;
        background: #18181b;
        padding: 6px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        box-sizing: border-box;
    `;
    const container = document.createElement("div");
    container.style.cssText = `
        position: relative;
        width: 100%;
        aspect-ratio: ${W} / ${H};
        user-select: none;
        background: #000;
        border-radius: 6px;
        overflow: hidden;
    `;
    wrap.appendChild(container);

    const img = document.createElement("img");
    img.src = model.get("image_url");
    img.style.cssText = `
        position: absolute; inset: 0; width: 100%; height: 100%;
        pointer-events: none;
    `;
    img.draggable = false;
    container.appendChild(img);

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.style.cssText = `
        position: absolute; inset: 0; width: 100%; height: 100%;
        cursor: crosshair; touch-action: none;
    `;
    svg.setAttribute("preserveAspectRatio", "none");
    container.appendChild(svg);

    el.appendChild(wrap);

    let shapes = [];
    let drawing = null;
    let drawingNode = null; // SVG node for the in-progress shape; updated in mousemove
    let pendingPoly = null;
    let hoverIndex = -1;
    let vbox = { x: 0, y: 0, w: W, h: H };
    const SNAP_PX = 10;
    const LASSO_MIN_PX = 1; // viewbox-px gate on lasso vert push

    function applyViewbox() {
        const sx = W / vbox.w;
        const sy = H / vbox.h;
        img.style.transformOrigin = "0 0";
        img.style.transform = `scale(${sx}, ${sy}) translate(${-vbox.x}px, ${-vbox.y}px)`;
        svg.setAttribute("viewBox", `${vbox.x} ${vbox.y} ${vbox.w} ${vbox.h}`);
    }
    applyViewbox();

    function setShapes(next) {
        if (next === shapes) return;
        if (next.length === 0 && shapes.length === 0) return;
        shapes = next;
        model.set("shapes", shapes);
        model.save_changes();
    }

    function popLastShape() {
        if (shapes.length === 0) return;
        setShapes(shapes.slice(0, -1));
    }

    function getXY(e) {
        const r = svg.getBoundingClientRect();
        const fx = (e.clientX - r.left) / r.width;
        const fy = (e.clientY - r.top) / r.height;
        return [vbox.x + fx * vbox.w, vbox.y + fy * vbox.h];
    }

    function vboxScalePerSvgPx() {
        return vbox.w / svg.getBoundingClientRect().width;
    }

    function makeEl(tag, attrs) {
        const n = document.createElementNS(svgNS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        return n;
    }

    function pointsAttr(verts) {
        return verts.map((v) => v.join(",")).join(" ");
    }

    function shapeNode(s, color, opts) {
        opts = opts || {};
        const common = {
            stroke: color,
            "stroke-width": opts.lw || 2,
            "vector-effect": "non-scaling-stroke",
            "stroke-dasharray": opts.dashed ? "6,4" : "",
        };
        const fillOp = opts.fillOp == null ? 0.15 : opts.fillOp;
        if (s.type === "rect") {
            const [x0, y0] = s.verts[0];
            const [x1, y1] = s.verts[2];
            return makeEl("rect", {
                ...common,
                x: Math.min(x0, x1),
                y: Math.min(y0, y1),
                width: Math.abs(x1 - x0),
                height: Math.abs(y1 - y0),
                fill: color,
                "fill-opacity": fillOp,
            });
        }
        if (s.type === "polygon") {
            return makeEl("polygon", {
                ...common,
                points: pointsAttr(s.verts),
                fill: color,
                "fill-opacity": fillOp,
            });
        }
        if (s.type === "polyline") {
            return makeEl("polyline", {
                ...common,
                points: pointsAttr(s.verts),
                fill: "none",
            });
        }
        return null;
    }

    function updateDrawingNode() {
        if (!drawing || !drawingNode) return;
        if (drawing.type === "rect") {
            const [x0, y0] = drawing.verts[0];
            const [x1, y1] = drawing.verts[2];
            drawingNode.setAttribute("x", Math.min(x0, x1));
            drawingNode.setAttribute("y", Math.min(y0, y1));
            drawingNode.setAttribute("width", Math.abs(x1 - x0));
            drawingNode.setAttribute("height", Math.abs(y1 - y0));
        } else {
            drawingNode.setAttribute("points", pointsAttr(drawing.verts));
        }
    }

    function distPx(a, b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    function shouldSnapClosePoly(e) {
        if (!pendingPoly || pendingPoly.verts.length < 3) return false;
        const r = svg.getBoundingClientRect();
        const fx = pendingPoly.verts[0][0];
        const fy = pendingPoly.verts[0][1];
        const cx = r.left + ((fx - vbox.x) / vbox.w) * r.width;
        const cy = r.top + ((fy - vbox.y) / vbox.h) * r.height;
        return distPx([e.clientX, e.clientY], [cx, cy]) <= SNAP_PX;
    }

    function attachShapeListeners(n, i) {
        n.style.cursor = "pointer";
        n.dataset.idx = String(i);
        n.addEventListener("mouseenter", () => {
            hoverIndex = i;
            redraw();
        });
        n.addEventListener("mouseleave", () => {
            if (hoverIndex === i) {
                hoverIndex = -1;
                redraw();
            }
        });
        n.addEventListener("click", (ev) => {
            if (ev.altKey) {
                const next = shapes.slice();
                next.splice(i, 1);
                hoverIndex = -1;
                setShapes(next);
                ev.stopPropagation();
            }
        });
    }

    function redraw() {
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        drawingNode = null;
        shapes.forEach((s, i) => {
            const isHover = i === hoverIndex;
            const n = shapeNode(s, isHover ? "#fb923c" : "#22d3ee", {
                lw: isHover ? 3 : 2,
                fillOp: isHover ? 0.25 : 0.15,
            });
            if (n) {
                attachShapeListeners(n, i);
                svg.appendChild(n);
            }
        });
        if (drawing) {
            const n = shapeNode(drawing, "#ec4899", { dashed: true });
            if (n) {
                n.style.pointerEvents = "none";
                svg.appendChild(n);
                drawingNode = n;
            }
        }
        if (pendingPoly && pendingPoly.verts.length > 0) {
            const px = vboxScalePerSvgPx();
            const rPx = 5 * px;
            pendingPoly.verts.forEach(([x, y], i) => {
                const c = makeEl("circle", {
                    cx: x,
                    cy: y,
                    r: i === 0 ? rPx * 1.3 : rPx,
                    fill: i === 0 ? "#facc15" : "#ec4899",
                    stroke: "white",
                    "stroke-width": 1.5 * px,
                    "vector-effect": "non-scaling-stroke",
                });
                c.style.pointerEvents = "none";
                svg.appendChild(c);
            });
        }
    }

    function commitPendingPolygon() {
        if (pendingPoly && pendingPoly.verts.length >= 3) {
            setShapes([
                ...shapes,
                { type: "polygon", verts: pendingPoly.verts },
            ]);
        }
        pendingPoly = null;
        drawing = null;
        redraw();
    }

    function vboxEq(a, b) {
        return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
    }

    function zoomAt(clientX, clientY, factor) {
        const r = svg.getBoundingClientRect();
        const fx = (clientX - r.left) / r.width;
        const fy = (clientY - r.top) / r.height;
        const px = vbox.x + fx * vbox.w;
        const py = vbox.y + fy * vbox.h;
        let newW = vbox.w / factor;
        let newH = vbox.h / factor;
        const minW = Math.max(5, W * 0.02);
        const minH = Math.max(5, H * 0.02);
        if (newW < minW) newW = minW;
        if (newH < minH) newH = minH;
        if (newW > W) {
            newW = W;
            newH = H;
        }
        const next = { x: px - fx * newW, y: py - fy * newH, w: newW, h: newH };
        clampVboxObj(next);
        if (vboxEq(vbox, next)) return;
        vbox = next;
        applyViewbox();
        redraw();
    }
    function panBy(dxClient, dyClient) {
        const r = svg.getBoundingClientRect();
        const next = {
            x: vbox.x - dxClient * (vbox.w / r.width),
            y: vbox.y - dyClient * (vbox.h / r.height),
            w: vbox.w,
            h: vbox.h,
        };
        clampVboxObj(next);
        if (vboxEq(vbox, next)) return;
        vbox = next;
        applyViewbox();
        redraw();
    }
    function clampVboxObj(v) {
        if (v.x < 0) v.x = 0;
        if (v.y < 0) v.y = 0;
        if (v.x + v.w > W) v.x = W - v.w;
        if (v.y + v.h > H) v.y = H - v.h;
    }
    function fitView() {
        const next = { x: 0, y: 0, w: W, h: H };
        if (vboxEq(vbox, next)) return;
        vbox = next;
        applyViewbox();
        redraw();
    }

    let panStart = null; // panning iff panStart !== null

    function onWheel(e) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
        zoomAt(e.clientX, e.clientY, factor);
    }

    function onMouseDown(e) {
        if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
            panStart = [e.clientX, e.clientY];
            svg.style.cursor = "grabbing";
            e.preventDefault();
            return;
        }
        if (e.button !== 0) return;
        svg.focus();
        const tool = model.get("tool");
        if (tool === "polygon" && shouldSnapClosePoly(e)) {
            commitPendingPolygon();
            e.preventDefault();
            return;
        }
        const [x, y] = getXY(e);
        if (tool === "rectangle") {
            drawing = {
                type: "rect",
                verts: [
                    [x, y],
                    [x, y],
                    [x, y],
                    [x, y],
                ],
            };
            redraw();
        } else if (tool === "lasso") {
            drawing = { type: "polygon", verts: [[x, y]] };
            redraw();
        } else if (tool === "polygon") {
            if (!pendingPoly) pendingPoly = { type: "polygon", verts: [] };
            pendingPoly.verts.push([x, y]);
            drawing = { type: "polyline", verts: pendingPoly.verts };
            redraw();
        }
        e.preventDefault();
    }

    function onMouseMove(e) {
        if (panStart !== null) {
            const dx = e.clientX - panStart[0];
            const dy = e.clientY - panStart[1];
            panStart = [e.clientX, e.clientY];
            panBy(dx, dy);
            return;
        }
        if (!drawing) return;
        const tool = model.get("tool");
        const [x, y] = getXY(e);
        if (tool === "rectangle") {
            const [x0, y0] = drawing.verts[0];
            drawing.verts = [
                [x0, y0],
                [x, y0],
                [x, y],
                [x0, y],
            ];
            updateDrawingNode();
        } else if (tool === "lasso") {
            const last = drawing.verts[drawing.verts.length - 1];
            if (distPx(last, [x, y]) >= LASSO_MIN_PX) {
                drawing.verts.push([x, y]);
                updateDrawingNode();
            }
        }
    }

    function onMouseUp(e) {
        if (panStart !== null) {
            panStart = null;
            svg.style.cursor = "crosshair";
            return;
        }
        const tool = model.get("tool");
        if (tool === "rectangle" && drawing) {
            const [[x0, y0], , [x1, y1]] = drawing.verts;
            if (Math.abs(x1 - x0) >= 2 && Math.abs(y1 - y0) >= 2) {
                setShapes([...shapes, { type: "rect", verts: drawing.verts }]);
            }
            drawing = null;
            redraw();
        } else if (tool === "lasso" && drawing && drawing.verts.length >= 3) {
            setShapes([...shapes, { type: "polygon", verts: drawing.verts }]);
            drawing = null;
            redraw();
        }
    }

    function onKeyDown(e) {
        const tool = model.get("tool");
        if (e.key === "r" || e.key === "R") {
            model.set("tool", "rectangle");
            model.save_changes();
            e.preventDefault();
            return;
        }
        if (e.key === "p" || e.key === "P") {
            model.set("tool", "polygon");
            model.save_changes();
            e.preventDefault();
            return;
        }
        if (e.key === "l" || e.key === "L") {
            model.set("tool", "lasso");
            model.save_changes();
            e.preventDefault();
            return;
        }
        if (e.key === "f" || e.key === "F") {
            fitView();
            e.preventDefault();
            return;
        }
        if (e.key === "Enter") {
            if (tool === "polygon" && pendingPoly) commitPendingPolygon();
            e.preventDefault();
            return;
        }
        if (e.key === "Escape") {
            pendingPoly = null;
            drawing = null;
            redraw();
            e.preventDefault();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
            popLastShape();
            e.preventDefault();
            return;
        }
        if (e.key === "Delete" || e.key === "Backspace") {
            popLastShape();
            e.preventDefault();
            return;
        }
    }

    svg.tabIndex = 0;
    svg.addEventListener("wheel", onWheel, { passive: false });
    svg.addEventListener("mousedown", onMouseDown);
    svg.addEventListener("mousemove", onMouseMove);
    svg.addEventListener("mouseup", onMouseUp);
    svg.addEventListener("mouseleave", (e) => {
        if (panStart === null) onMouseUp(e);
    });
    svg.addEventListener("keydown", onKeyDown);
    svg.addEventListener("contextmenu", (e) => e.preventDefault());

    function updateCursor() {
        svg.style.cursor = "crosshair";
        svg.title = `Tool: ${model.get("tool")}. R/P/L: tools. Enter: close poly. Esc: cancel. Ctrl+Z: undo. Alt+click shape: delete. Wheel: zoom. Shift+drag: pan. F: fit.`;
    }
    updateCursor();

    model.on("change:tool", () => {
        const hadInProgress = pendingPoly !== null || drawing !== null;
        pendingPoly = null;
        drawing = null;
        updateCursor();
        if (hadInProgress) redraw();
    });
    model.on("change:clear_trigger", () => {
        const hadInProgress = drawing !== null || pendingPoly !== null;
        drawing = null;
        pendingPoly = null;
        if (shapes.length === 0 && !hadInProgress) return;
        setShapes([]);
        redraw();
    });
    model.on("change:close_poly_trigger", () => {
        commitPendingPolygon();
    });
    model.on("change:undo_trigger", () => {
        popLastShape();
    });
    model.on("change:fit_trigger", () => {
        fitView();
    });
}

export default { render };
