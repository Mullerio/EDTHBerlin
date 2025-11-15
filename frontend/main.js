const form = document.getElementById("grid-form");
const rowsInput = document.getElementById("rows-input");
const colsInput = document.getElementById("cols-input");
const cellSizeInput = document.getElementById("cell-size-input");
const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");
const resetButton = document.getElementById("reset-shape");
const closeButton = document.getElementById("close-shape");
const fillButton = document.getElementById("fill-grid");
const statusEl = document.getElementById("status");
const zoomInput = document.getElementById("zoom-input");
const zoomLabel = document.getElementById("zoom-label");
const centroidGridEl = document.getElementById("centroid-grid");
const centroidMetersEl = document.getElementById("centroid-meters");
const canvasWrapper = document.querySelector(".canvas-wrapper");
const attackerSelect = document.getElementById("attacker-select");
const addAttackerButton = document.getElementById("add-attacker");
const removeAttackerButton = document.getElementById("remove-attacker");
const assignStartButton = document.getElementById("assign-start");
const assignTargetButton = document.getElementById("assign-target");
const attackerListEl = document.getElementById("attacker-list");
const copyPythonButton = document.getElementById("copy-python");
const downloadJsonButton = document.getElementById("download-json");
const pythonPreview = document.getElementById("python-preview");
const shapeCountEl = document.getElementById("shape-count");
const shapeListEl = document.getElementById("shape-list");
const loadCsvButton = document.getElementById("load-csv-grid");
const csvGridInput = document.getElementById("csv-grid-input");

const MAX_CANVAS_SIZE = 640;
const MIN_CELL_SIZE = 6;

const initialCellSizeMeters = sanitizeCellSizeMeters(
  Number(cellSizeInput ? cellSizeInput.value : 1)
);
if (cellSizeInput) {
  cellSizeInput.value = initialCellSizeMeters;
}
const initialZoomPercent = zoomInput ? Number(zoomInput.value) : 100;
const state = {
  rows: parseInt(rowsInput.value, 10),
  cols: parseInt(colsInput.value, 10),
  cellSize: 32,
  vertices: [],
  grid: [],
  hasGrid: false,
  completedShapes: [],
  shapeSequence: 0,
  attackers: [],
  activeAttackerId: null,
  attackerSequence: 0,
  selectionMode: null,
  cellSizeMeters: initialCellSizeMeters,
  zoom: Number.isFinite(initialZoomPercent) ? initialZoomPercent / 100 : 1,
};

init();

function init() {
  form.addEventListener("submit", handleGridSubmit);
  canvas.addEventListener("click", handleCanvasClick);
  resetButton.addEventListener("click", resetShape);
  closeButton.addEventListener("click", closeShape);
  fillButton.addEventListener("click", fillGridWithPolygon);
  if (cellSizeInput) {
    cellSizeInput.addEventListener("input", handleCellSizeMetersChange);
  }
  if (zoomInput) {
    zoomInput.addEventListener("input", handleZoomChange);
  }
  if (canvasWrapper) {
    canvasWrapper.addEventListener("wheel", handleWheelZoom, { passive: false });
  }
  attackerSelect.addEventListener("change", handleAttackerSelectChange);
  addAttackerButton.addEventListener("click", addAttacker);
  removeAttackerButton.addEventListener("click", removeAttacker);
  assignStartButton.addEventListener("click", () => enableSelectionMode("start"));
  assignTargetButton.addEventListener("click", () => enableSelectionMode("target"));
  copyPythonButton.addEventListener("click", copyPythonSnippet);
  downloadJsonButton.addEventListener("click", downloadJsonPayload);
  if (loadCsvButton && csvGridInput) {
    loadCsvButton.addEventListener("click", () => csvGridInput.click());
    csvGridInput.addEventListener("change", handleCsvGridUpload);
  }
  window.addEventListener("resize", drawGrid);

  createGrid(state.rows, state.cols);
  applyZoom();
  updateZoomLabel();
}

function handleGridSubmit(event) {
  event.preventDefault();
  const rows = clamp(
    parseInt(rowsInput.value, 10),
    Number(rowsInput.min),
    Number(rowsInput.max)
  );
  const cols = clamp(
    parseInt(colsInput.value, 10),
    Number(colsInput.min),
    Number(colsInput.max)
  );
  const cellMeters = sanitizeCellSizeMeters(
    Number(cellSizeInput ? cellSizeInput.value : state.cellSizeMeters)
  );

  rowsInput.value = rows;
  colsInput.value = cols;
  if (cellSizeInput) {
    cellSizeInput.value = cellMeters;
  }
  state.cellSizeMeters = cellMeters;
  createGrid(rows, cols);
}

function createGrid(rows, cols) {
  state.rows = rows;
  state.cols = cols;
  state.cellSize = computeCellSize(rows, cols);
  canvas.width = state.cols * state.cellSize + 1;
  canvas.height = state.rows * state.cellSize + 1;
  state.grid = buildGrid(rows, cols, 1);
  state.vertices = [];
  state.hasGrid = true;
  state.completedShapes = [];
  state.shapeSequence = 0;
  state.selectionMode = null;

  initializeAttackers();
  updatePythonPreview();
  updateShapeSummary();
  updateStatus("Grid created. Click on the canvas to add vertices.");
  drawGrid();
  applyZoom();
}

async function handleCsvGridUpload(event) {
  const input = event?.target ?? csvGridInput;
  if (!input || !input.files || input.files.length === 0) {
    return;
  }
  const file = input.files[0];
  try {
    updateStatus(`Loading ${file.name || "CSV"}...`);
    const text = await file.text();
    const matrix = parseCsvGrid(text);
    applyImportedGrid(matrix, file.name);
  } catch (error) {
    console.error("CSV grid import failed:", error);
    const message =
      error && typeof error.message === "string"
        ? error.message
        : "Unexpected error.";
    updateStatus(`CSV import failed: ${message}`);
  } finally {
    input.value = "";
  }
}

function applyImportedGrid(matrix, sourceName) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new Error("CSV grid is empty.");
  }
  const cols = matrix[0]?.length ?? 0;
  if (cols === 0) {
    throw new Error("CSV grid rows have no values.");
  }

  const safeMatrix = matrix.map((row) => row.slice());
  const rows = safeMatrix.length;

  rowsInput.value = String(rows);
  colsInput.value = String(cols);

  createGrid(rows, cols);
  state.grid = safeMatrix;
  drawGrid();
  updatePythonPreview();
  updateStatus(
    sourceName
      ? `Grid imported from "${sourceName}" (${rows}x${cols}). Assign attackers next.`
      : `Grid imported (${rows}x${cols}). Assign attackers next.`
  );
}

function parseCsvGrid(csvText) {
  if (typeof csvText !== "string" || csvText.trim().length === 0) {
    throw new Error("CSV file is empty.");
  }
  const sanitized = csvText.replace(/^\uFEFF/, "");
  const lines = sanitized
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length === 0) {
    throw new Error("CSV file is empty.");
  }

  let expectedCols = null;
  const matrix = lines.map((line, rowIndex) => {
    const tokens = line
      .split(/[,;\s]+/)
      .map((token) => token.trim())
      .filter((token) => token.length > 0);

    if (tokens.length === 0) {
      throw new Error(`Row ${rowIndex + 1} has no values.`);
    }

    if (expectedCols === null) {
      expectedCols = tokens.length;
    } else if (tokens.length !== expectedCols) {
      throw new Error(
        `Row ${rowIndex + 1} has ${tokens.length} columns; expected ${expectedCols}.`
      );
    }

    return tokens.map((token, colIndex) => {
      if (!/^-?\d+$/.test(token)) {
        throw new Error(
          `Row ${rowIndex + 1}, column ${colIndex + 1} is not an integer.`
        );
      }
      const numeric = Number(token);
      if (numeric !== 0 && numeric !== 1) {
        throw new Error(
          `Row ${rowIndex + 1}, column ${colIndex + 1} must be 0 or 1.`
        );
      }
      return numeric;
    });
  });

  return matrix;
}

function handleCanvasClick(event) {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }

  const { x, y } = getCanvasCoordinates(event);
  const cell = getCellFromCoordinates(x, y);

  if (state.selectionMode) {
    assignCellToAttacker(cell.row, cell.col);
    return;
  }

  const snapped = snapToGrid(x, y);
  const firstVertex = state.vertices[0];

  if (
    state.vertices.length >= 3 &&
    firstVertex &&
    isSamePoint(snapped, firstVertex)
  ) {
    const finalized = finalizeCurrentShape();
    if (finalized) {
      updateStatus(
        "Shape stored. Continue drawing to add another shape or fill the grid."
      );
    }
    return;
  }

  const lastVertex = state.vertices[state.vertices.length - 1];
  if (lastVertex && lastVertex.x === snapped.x && lastVertex.y === snapped.y) {
    updateStatus("Vertex already placed there. Choose a different position.");
    return;
  }

  state.vertices.push(snapped);
  drawGrid();
  updateStatus(
    state.vertices.length >= 3
      ? "Click Close shape or click the first vertex again to store this shape."
      : "Add at least three points."
  );
}

function closeShape() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  if (state.vertices.length < 3) {
    updateStatus("Need at least three vertices to close the shape.");
    return;
  }
  const finalized = finalizeCurrentShape();
  if (finalized) {
    updateStatus("Shape stored. Continue drawing or fill the grid.");
  }
}

function resetShape() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }

  state.vertices = [];
  state.completedShapes = [];
  state.shapeSequence = 0;
  state.grid = buildGrid(state.rows, state.cols, 1);
  drawGrid();
  updatePythonPreview();
  updateShapeSummary();
  updateStatus("Shapes reset. Start adding vertices again.");
}

function fillGridWithPolygon() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  if (state.completedShapes.length === 0) {
    updateStatus("Close at least one shape before filling the grid.");
    return;
  }

  const filledGrid = buildGrid(state.rows, state.cols, 1);
  const polygons = state.completedShapes.map((shape) => shape.vertices);

  for (let row = 0; row < state.rows; row += 1) {
    for (let col = 0; col < state.cols; col += 1) {
      const center = {
        x: col * state.cellSize + state.cellSize / 2,
        y: row * state.cellSize + state.cellSize / 2,
      };
      if (
        polygons.some((polygon) => isPointInPolygon(center, polygon))
      ) {
        filledGrid[row][col] = 0;
      }
    }
  }

  state.grid = filledGrid;
  drawGrid();
  updatePythonPreview();
  updateStatus(
    `Grid filled. 0 = inside any of the ${state.completedShapes.length} stored shape(s). Click "Download JSON" to export.`
  );
}

function drawGrid() {
  if (!state.hasGrid) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCells();
  drawGridLines();
  drawCompletedShapes();
  drawActiveShape();
  drawShapeCentroids();
  drawAttackerMarkers();
}

function drawCells() {
  for (let row = 0; row < state.rows; row += 1) {
    for (let col = 0; col < state.cols; col += 1) {
      const value = state.grid[row][col];
      ctx.fillStyle =
        value === 0
          ? "rgba(74, 201, 155, 0.65)"
          : "rgba(28, 39, 63, 0.08)";
      ctx.fillRect(
        col * state.cellSize,
        row * state.cellSize,
        state.cellSize,
        state.cellSize
      );
    }
  }
}

function drawGridLines() {
  ctx.strokeStyle = "rgba(35, 49, 71, 0.45)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= state.cols; c += 1) {
    const x = c * state.cellSize + 0.5;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
  }
  for (let r = 0; r <= state.rows; r += 1) {
    const y = r * state.cellSize + 0.5;
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
  }
  ctx.stroke();
}

function drawCompletedShapes() {
  if (state.completedShapes.length === 0) return;
  state.completedShapes.forEach((shape) => {
    drawShapePath(shape.vertices, { stroke: "#2563eb", closePath: true });
  });
}

function drawActiveShape() {
  if (state.vertices.length === 0) return;
  drawShapePath(state.vertices, { stroke: "#ff7b00", closePath: false });
}

function drawShapePath(vertices, { stroke = "#0f7bff", closePath = true } = {}) {
  if (!vertices || vertices.length === 0) return;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(vertices[0].x, vertices[0].y);
  for (let i = 1; i < vertices.length; i += 1) {
    ctx.lineTo(vertices[i].x, vertices[i].y);
  }
  if (closePath && vertices.length > 2) {
    ctx.lineTo(vertices[0].x, vertices[0].y);
  }
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#111827";
  vertices.forEach((vertex) => {
    ctx.beginPath();
    ctx.arc(vertex.x, vertex.y, Math.max(4, state.cellSize * 0.08), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function drawShapeCentroids() {
  if (state.completedShapes.length === 0) return;
  state.completedShapes.forEach((shape) => {
    const pixel = shape.centroid?.pixel;
    if (!pixel) return;
    const radius = Math.max(5, state.cellSize * 0.18);

    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#0f172a";
    ctx.fillStyle = "#14b8a6";
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(pixel.x - radius * 1.2, pixel.y);
    ctx.lineTo(pixel.x + radius * 1.2, pixel.y);
    ctx.moveTo(pixel.x, pixel.y - radius * 1.2);
    ctx.lineTo(pixel.x, pixel.y + radius * 1.2);
    ctx.stroke();
    ctx.restore();
  });
}

function drawAttackerMarkers() {
  if (!state.hasGrid || state.attackers.length === 0) return;

  const radius = Math.max(6, state.cellSize * 0.2);
  ctx.lineWidth = 2;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `bold ${Math.max(12, state.cellSize * 0.35)}px system-ui`;

  state.attackers.forEach((attacker, index) => {
    const suffix = index + 1;
    if (attacker.start) {
      drawMarkerForCell(attacker.start, {
        color: "#22c55e",
        label: `S${suffix}`,
        isActive: attacker.id === state.activeAttackerId,
      });
    }
    if (attacker.target) {
      drawMarkerForCell(attacker.target, {
        color: "#ef4444",
        label: `T${suffix}`,
        isActive: attacker.id === state.activeAttackerId,
      });
    }
  });
}

function getCanvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function snapToGrid(x, y) {
  const snapX = clamp(
    Math.round(x / state.cellSize) * state.cellSize,
    0,
    state.cols * state.cellSize
  );
  const snapY = clamp(
    Math.round(y / state.cellSize) * state.cellSize,
    0,
    state.rows * state.cellSize
  );
  return { x: snapX, y: snapY };
}

function getCellFromCoordinates(x, y) {
  const col = clamp(Math.floor(x / state.cellSize), 0, state.cols - 1);
  const row = clamp(Math.floor(y / state.cellSize), 0, state.rows - 1);
  return { row, col };
}

function cellToPixelCenter(cell) {
  return {
    x: cell.col * state.cellSize + state.cellSize / 2,
    y: cell.row * state.cellSize + state.cellSize / 2,
  };
}

function drawMarkerForCell(cell, { color, label, isActive }) {
  const center = cellToPixelCenter(cell);
  const radius = Math.max(6, state.cellSize * 0.2);

  ctx.save();
  ctx.globalAlpha = isActive ? 1 : 0.75;
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(15, 23, 42, 0.85)";
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, center.x, center.y);
  ctx.restore();
}

function isSamePoint(a, b) {
  return Boolean(a && b) && a.x === b.x && a.y === b.y;
}

function finalizeCurrentShape() {
  if (state.vertices.length < 3) {
    return false;
  }
  const centroidPixel = computePolygonCentroid(state.vertices);
  if (!centroidPixel) {
    return false;
  }
  const centroidGrid = {
    row: centroidPixel.y / state.cellSize,
    col: centroidPixel.x / state.cellSize,
  };
  state.completedShapes.push({
    id: ++state.shapeSequence,
    vertices: cloneVertices(state.vertices),
    centroid: {
      pixel: centroidPixel,
      grid: centroidGrid,
    },
  });
  state.vertices = [];
  updateShapeSummary();
  updatePythonPreview();
  drawGrid();
  return true;
}

function cloneVertices(vertices) {
  return vertices.map((vertex) => ({ x: vertex.x, y: vertex.y }));
}

function computePolygonCentroid(points) {
  if (!points || points.length < 3) return null;
  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const cross = current.x * next.y - next.x * current.y;
    area += cross;
    cx += (current.x + next.x) * cross;
    cy += (current.y + next.y) * cross;
  }
  area *= 0.5;
  if (Math.abs(area) < 1e-6) return null;
  cx /= 6 * area;
  cy /= 6 * area;
  return { x: cx, y: cy };
}

function updateShapeSummary() {
  const count = state.completedShapes.length;
  if (shapeCountEl) {
    shapeCountEl.textContent = String(count);
  }
  if (!centroidGridEl || !centroidMetersEl) return;
  if (count === 0) {
    centroidGridEl.textContent = "—";
    centroidMetersEl.textContent = "—";
  } else {
    const lastShape = state.completedShapes[count - 1];
    const meters = gridToMeters(lastShape.centroid.grid);
    centroidGridEl.textContent = `row ${formatDecimal(
      lastShape.centroid.grid.row
    )}, col ${formatDecimal(lastShape.centroid.grid.col)}`;
    centroidMetersEl.textContent = `${formatDecimal(
      meters.x
    )} m, ${formatDecimal(meters.y)} m`;
  }
  if (shapeListEl) {
    shapeListEl.innerHTML = "";
    if (count === 0) {
      const li = document.createElement("li");
      li.classList.add("shape-list__empty");
      li.textContent = "No shapes yet. Close a shape to store it.";
      shapeListEl.append(li);
    } else {
      state.completedShapes.forEach((shape) => {
        const meters = gridToMeters(shape.centroid.grid);
        const li = document.createElement("li");
        li.textContent = `Shape ${shape.id}: row ${formatDecimal(
          shape.centroid.grid.row
        )}, col ${formatDecimal(shape.centroid.grid.col)} | ${formatDecimal(
          meters.x
        )}m, ${formatDecimal(meters.y)}m`;
        li.title = `Vertices: ${shape.vertices.length}`;
        shapeListEl.append(li);
      });
    }
  }
}

function gridToMeters(grid) {
  if (!grid) return { x: 0, y: 0 };
  return {
    x: grid.col * state.cellSizeMeters,
    y: grid.row * state.cellSizeMeters,
  };
}

function cellToMeters(cell) {
  if (!cell) return null;
  const grid = { row: cell.row, col: cell.col };
  return gridToMeters(grid);
}

function getShapesExportData() {
  return state.completedShapes.map((shape) => {
    const centroidMeters = gridToMeters(shape.centroid.grid);
    const centroid = {
      grid: {
        row: shape.centroid.grid.row,
        col: shape.centroid.grid.col,
      },
      meters: centroidMeters,
    };
    const vertices = shape.vertices.map((vertex) => {
      const grid = {
        row: vertex.y / state.cellSize,
        col: vertex.x / state.cellSize,
      };
      return {
        grid,
        meters: gridToMeters(grid),
      };
    });
    return {
      id: shape.id,
      centroid,
      vertices,
    };
  });
}

function cellToPayload(cell) {
  if (!cell) return null;
  const grid = {
    row: cell.row,
    col: cell.col,
  };
  return {
    grid,
    meters: gridToMeters(grid),
  };
}

function buildGrid(rows, cols, fillValue) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => fillValue)
  );
}

function computeCellSize(rows, cols) {
  const largestDimension = Math.max(rows, cols);
  const rawSize = Math.floor(MAX_CANVAS_SIZE / largestDimension);
  return Math.max(rawSize, MIN_CELL_SIZE);
}

function clamp(value, min, max) {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function sanitizeCellSizeMeters(value) {
  const minAttr = cellSizeInput ? Number(cellSizeInput.min) : 0.01;
  const maxAttr = cellSizeInput ? Number(cellSizeInput.max) : 1000;
  const min = Number.isFinite(minAttr) && minAttr > 0 ? minAttr : 0.01;
  const max =
    Number.isFinite(maxAttr) && maxAttr > min ? maxAttr : Math.max(min * 10, 1000);
  const numeric = Number.isFinite(value) ? value : min;
  return clamp(numeric, min, max);
}

function updateStatus(message) {
  statusEl.textContent = message;
}

function getZoomPercent() {
  return Math.round((state.zoom || 1) * 100);
}

function setZoomPercent(percent, anchor) {
  if (!Number.isFinite(percent)) return;
  const min = zoomInput ? Number(zoomInput.min) : 25;
  const max = zoomInput ? Number(zoomInput.max) : 400;
  const normalized = clamp(percent, min, max);
  const previousScale = state.zoom || 1;
  const nextScale = normalized / 100;
  state.zoom = nextScale;
  if (zoomInput && Number.isFinite(normalized)) {
    zoomInput.value = String(normalized);
  }
  updateZoomLabel(normalized);
  applyZoom(anchor, previousScale, nextScale);
}

function handleZoomChange(event) {
  const value = Number(event.target.value);
  setZoomPercent(value, getWrapperCenterAnchor());
}

function handleWheelZoom(event) {
  if (!event.ctrlKey && !event.metaKey) {
    return;
  }
  event.preventDefault();
  const step = zoomInput ? Number(zoomInput.step || 5) : 5;
  const current = getZoomPercent();
  const direction = Math.sign(event.deltaY || 0) || 1;
  const next = current - direction * step;
  setZoomPercent(next, {
    clientX: event.clientX,
    clientY: event.clientY,
  });
}

function applyZoom(anchor, prevScale, nextScale) {
  const scale = Number.isFinite(nextScale) ? nextScale : state.zoom || 1;
  const previous = Number.isFinite(prevScale) ? prevScale : scale || 1;
  canvas.style.transform = `scale(${scale})`;
  if (!anchor || !canvasWrapper || previous === 0) {
    return;
  }
  const rect = canvasWrapper.getBoundingClientRect();
  const relativeX = anchor.clientX - rect.left;
  const relativeY = anchor.clientY - rect.top;
  const scrollLeft = canvasWrapper.scrollLeft;
  const scrollTop = canvasWrapper.scrollTop;
  const ratio = scale / previous;
  const newScrollLeft = (relativeX + scrollLeft) * ratio - relativeX;
  const newScrollTop = (relativeY + scrollTop) * ratio - relativeY;
  canvasWrapper.scrollTo({
    left: newScrollLeft,
    top: newScrollTop,
  });
}

function updateZoomLabel(explicitValue) {
  if (!zoomLabel) return;
  let value = explicitValue;
  if (!Number.isFinite(value)) {
    if (zoomInput && Number.isFinite(Number(zoomInput.value))) {
      value = Number(zoomInput.value);
    } else {
      value = Math.round((state.zoom || 1) * 100);
    }
  }
  zoomLabel.textContent = `${Math.round(value)}%`;
}

function getWrapperCenterAnchor() {
  if (!canvasWrapper) return null;
  const rect = canvasWrapper.getBoundingClientRect();
  return {
    clientX: rect.left + rect.width / 2,
    clientY: rect.top + rect.height / 2,
  };
}

function handleCellSizeMetersChange(event) {
  const target = event?.target ?? cellSizeInput;
  if (!target) return;
  const normalized = sanitizeCellSizeMeters(Number(target.value));
  state.cellSizeMeters = normalized;
  target.value = normalized;
  updateShapeSummary();
  updatePythonPreview();
}

function initializeAttackers() {
  state.attackers = [];
  state.activeAttackerId = null;
  state.attackerSequence = 0;
  state.selectionMode = null;
  addAttacker();
}

function addAttacker() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  state.attackerSequence += 1;
  const attacker = {
    id: `att-${state.attackerSequence}`,
    name: `Attacker ${state.attackerSequence}`,
    start: null,
    target: null,
  };
  state.attackers.push(attacker);
  state.activeAttackerId = attacker.id;
  state.selectionMode = null;
  updateAttackerUI();
  drawGrid();
  updateStatus(`${attacker.name} added. Assign start/target cells.`);
}

function removeAttacker() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  if (state.attackers.length <= 1) {
    updateStatus("Keep at least one attacker in the scenario.");
    return;
  }
  const removedId = state.activeAttackerId;
  state.attackers = state.attackers.filter((attacker) => attacker.id !== removedId);
  const nextAttacker = state.attackers[state.attackers.length - 1];
  state.activeAttackerId = nextAttacker ? nextAttacker.id : null;
  state.selectionMode = null;
  updateAttackerUI();
  drawGrid();
  updateStatus("Attacker removed.");
}

function handleAttackerSelectChange(event) {
  setActiveAttacker(event.target.value);
}

function setActiveAttacker(attackerId) {
  state.activeAttackerId = attackerId;
  state.selectionMode = null;
  updateSelectionButtons();
  updateStatus(`Selected ${getActiveAttacker()?.name ?? "attacker"}.`);
}

function getActiveAttacker() {
  return state.attackers.find((attacker) => attacker.id === state.activeAttackerId);
}

function enableSelectionMode(mode) {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  const attacker = getActiveAttacker();
  if (!attacker) {
    updateStatus("Add at least one attacker.");
    return;
  }
  if (state.selectionMode === mode) {
    state.selectionMode = null;
    updateSelectionButtons();
    updateStatus("Placement mode cleared.");
    return;
  }
  state.selectionMode = mode;
  updateSelectionButtons();
  updateStatus(`Click a cell to set the ${mode} for ${attacker.name}.`);
}

function assignCellToAttacker(row, col) {
  const attacker = getActiveAttacker();
  if (!attacker || !state.selectionMode) {
    return;
  }

  const key = state.selectionMode === "start" ? "start" : "target";
  attacker[key] = { row, col };
  state.selectionMode = null;
  updateAttackerUI();
  drawGrid();
  updateStatus(`${attacker.name} ${key} set to (${row}, ${col}).`);
}

function updateAttackerUI() {
  if (!attackerSelect || !attackerListEl) return;

  attackerSelect.innerHTML = "";
  state.attackers.forEach((attacker) => {
    const option = document.createElement("option");
    option.value = attacker.id;
    option.textContent = attacker.name;
    if (attacker.id === state.activeAttackerId) {
      option.selected = true;
    }
    attackerSelect.append(option);
  });

  attackerSelect.disabled = state.attackers.length === 0;
  removeAttackerButton.disabled = state.attackers.length <= 1;
  const hasAttacker = state.attackers.length > 0;
  assignStartButton.disabled = !hasAttacker;
  assignTargetButton.disabled = !hasAttacker;

  attackerListEl.innerHTML = "";
  state.attackers.forEach((attacker) => {
    const li = document.createElement("li");
    li.textContent = `${attacker.name} — start: ${formatCellForDisplay(
      attacker.start
    )} | target: ${formatCellForDisplay(attacker.target)}`;
    if (attacker.id === state.activeAttackerId) {
      li.classList.add("is-active");
    }
    attackerListEl.append(li);
  });

  updateSelectionButtons();
  updatePythonPreview();
}

function formatCellForDisplay(cell) {
  if (!cell) return "unset";
  const meters = cellToMeters(cell);
  return `(${cell.row}, ${cell.col}) | ${meters ? `${formatDecimal(meters.x)}m, ${formatDecimal(meters.y)}m` : "—"}`;
}

function formatDecimal(value, digits = 2) {
  if (!Number.isFinite(value)) return "—";
  return Number(value.toFixed(digits)).toString();
}

function updateSelectionButtons() {
  if (!assignStartButton || !assignTargetButton) return;
  assignStartButton.classList.toggle("is-active", state.selectionMode === "start");
  assignTargetButton.classList.toggle(
    "is-active",
    state.selectionMode === "target"
  );
}

function updatePythonPreview() {
  if (!pythonPreview) return;
  if (!state.hasGrid || state.grid.length === 0) {
    pythonPreview.value = "No grid yet. Create and fill a grid to see the snippet.";
    return;
  }
  pythonPreview.value = buildPythonSnippet();
}

function buildPythonSnippet() {
  const matrixRows = state.grid
    .map((row) => `    [${row.join(", ")}]`)
    .join(",\n");

  const attackerLines = state.attackers.map((attacker) => formatAttackerForPython(attacker)).join("\n");

  const shapes = getShapesExportData();
  const legacyCentroid = shapes[0]?.centroid ?? null;
  const centroidBlock = legacyCentroid
    ? [
        "shape_centroid = {",
        `    "grid": {"row": ${formatNumberForPython(legacyCentroid.grid.row)}, "col": ${formatNumberForPython(
          legacyCentroid.grid.col
        )}},`,
        `    "meters": {"x": ${formatNumberForPython(legacyCentroid.meters.x)}, "y": ${formatNumberForPython(
          legacyCentroid.meters.y
        )}},`,
        "}",
      ]
    : ["shape_centroid = None"];

  const vertexData = shapes[0]?.vertices ?? [];
  const vertexBlock =
    vertexData.length > 0
      ? ["shape_vertices = [", ...vertexData.map((vertex) => formatVertexForPython(vertex)), "]"]
      : ["shape_vertices = []"];

  const shapesBlock =
    shapes.length > 0
      ? ["shapes = [", ...shapes.map((shape) => formatShapeForPython(shape)), "]"]
      : ["shapes = []"];

  return [
    "import numpy as np",
    "",
    `cell_size_m = ${formatNumberForPython(state.cellSizeMeters)}`,
    "",
    "matrix = np.array([",
    matrixRows,
    "], dtype=int)",
    "",
    ...centroidBlock,
    "",
    ...vertexBlock,
    "",
    ...shapesBlock,
    "",
    "attackers = [",
    attackerLines || "    # Add attackers to populate this list",
    "]",
    "",
    "print(matrix.shape)",
    "print(attackers)",
  ].join("\n");
}

function formatNumberForPython(value, digits = 4) {
  if (!Number.isFinite(value)) return "None";
  return Number(value.toFixed(digits)).toString();
}

function formatVertexForPython(vertex, indent = "    ") {
  const row = formatNumberForPython(vertex.grid.row);
  const col = formatNumberForPython(vertex.grid.col);
  const x = formatNumberForPython(vertex.meters.x);
  const y = formatNumberForPython(vertex.meters.y);
  return `${indent}{"grid": {"row": ${row}, "col": ${col}}, "meters": {"x": ${x}, "y": ${y}}},`;
}

function formatCellPayloadForPython(cell) {
  const payload = cellToPayload(cell);
  if (!payload) return "None";
  const row = formatNumberForPython(payload.grid.row);
  const col = formatNumberForPython(payload.grid.col);
  const x = formatNumberForPython(payload.meters.x);
  const y = formatNumberForPython(payload.meters.y);
  return `{"grid": {"row": ${row}, "col": ${col}}, "meters": {"x": ${x}, "y": ${y}}}`;
}

function formatAttackerForPython(attacker) {
  const safeName = JSON.stringify(attacker.name);
  const start = formatCellPayloadForPython(attacker.start);
  const target = formatCellPayloadForPython(attacker.target);
  return [
    `    {`,
    `        "name": ${safeName},`,
    `        "start": ${start},`,
    `        "target": ${target},`,
    `    },`,
  ].join("\n");
}

function formatShapeForPython(shape) {
  const centroidRow = formatNumberForPython(shape.centroid.grid.row);
  const centroidCol = formatNumberForPython(shape.centroid.grid.col);
  const centroidX = formatNumberForPython(shape.centroid.meters.x);
  const centroidY = formatNumberForPython(shape.centroid.meters.y);
  const vertexLines =
    shape.vertices.length > 0
      ? shape.vertices.map((vertex) => formatVertexForPython(vertex, "            "))
      : ["            # No vertices"];

  return [
    "    {",
    `        "id": ${shape.id},`,
    '        "centroid": {',
    `            "grid": {"row": ${centroidRow}, "col": ${centroidCol}},`,
    `            "meters": {"x": ${centroidX}, "y": ${centroidY}},`,
    "        },",
    '        "vertices": [',
    ...vertexLines,
    "        ],",
    "    },",
  ].join("\n");
}

async function copyPythonSnippet() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  const snippet = buildPythonSnippet();
  try {
    await navigator.clipboard.writeText(snippet);
    updateStatus("Python snippet copied to clipboard.");
  } catch (error) {
    if (pythonPreview) {
      pythonPreview.select();
      pythonPreview.setSelectionRange(0, snippet.length);
    }
    let success = false;
    try {
      success = document.execCommand("copy");
    } catch (commandError) {
      success = false;
    }
    updateStatus(
      success
        ? "Clipboard access blocked, but the snippet is copied."
        : "Clipboard blocked. Snippet selected — copy manually."
    );
  }
}

async function downloadJsonPayload() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  const payload = buildJsonPayload();
  const jsonString = JSON.stringify(payload, null, 2);

  const savedViaPicker = await trySaveWithFileSystemAccess(jsonString);
  if (savedViaPicker) {
    updateStatus("JSON saved via file picker.");
    return;
  }

  fallbackDownload(jsonString);
  updateStatus("JSON downloaded (browser default).");
}

function buildJsonPayload() {
  const shapes = getShapesExportData();
  const legacyCentroid = shapes[0]?.centroid ?? null;
  const legacyVertices = shapes[0]?.vertices ?? [];
  return {
    rows: state.rows,
    cols: state.cols,
    cellSizeMeters: state.cellSizeMeters,
    centroid: legacyCentroid,
    shapeVertices: legacyVertices,
    shapes,
    matrix: state.grid,
    attackers: state.attackers.map((attacker) => ({
      name: attacker.name,
      start: cellToPayload(attacker.start),
      target: cellToPayload(attacker.target),
    })),
    generatedAt: new Date().toISOString(),
  };
}

async function trySaveWithFileSystemAccess(jsonString) {
  if (!window.showSaveFilePicker) {
    return false;
  }
  try {
    const handle = await window.showSaveFilePicker({
      suggestedName: `grid-${state.rows}x${state.cols}.json`,
      types: [
        {
          description: "JSON Files",
          accept: { "application/json": [".json"] },
        },
      ],
    });
    const writable = await handle.createWritable();
    await writable.write(jsonString);
    await writable.close();
    return true;
  } catch (error) {
    if (error.name !== "AbortError") {
      console.warn("File System Access API save failed:", error);
    }
    return false;
  }
}

function fallbackDownload(jsonString) {
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `grid-${state.rows}x${state.cols}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function isPointInPolygon(point, polygon) {
  if (polygon.length < 3) return false;
  if (isOnPolygonBoundary(point, polygon)) return true;

  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersects =
      yi > point.y !== yj > point.y &&
      point.x <
        ((xj - xi) * (point.y - yi)) / (yj - yi + Number.EPSILON) + xi;

    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function isOnPolygonBoundary(point, polygon) {
  for (let i = 0; i < polygon.length; i += 1) {
    const start = polygon[i];
    const end = polygon[(i + 1) % polygon.length];
    if (isPointOnSegment(point, start, end)) {
      return true;
    }
  }
  return false;
}

function isPointOnSegment(point, start, end) {
  const cross =
    (point.y - start.y) * (end.x - start.x) -
    (point.x - start.x) * (end.y - start.y);
  if (Math.abs(cross) > 1e-6) return false;

  const dot =
    (point.x - start.x) * (end.x - start.x) +
    (point.y - start.y) * (end.y - start.y);
  if (dot < 0) return false;

  const squaredLength =
    (end.x - start.x) * (end.x - start.x) +
    (end.y - start.y) * (end.y - start.y);
  return dot <= squaredLength;
}

