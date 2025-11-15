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
  shapeClosed: false,
  grid: [],
  hasGrid: false,
  attackers: [],
  activeAttackerId: null,
  attackerSequence: 0,
  selectionMode: null,
  cellSizeMeters: initialCellSizeMeters,
  shapeCentroid: null,
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
  state.shapeClosed = false;
  state.hasGrid = true;
  state.selectionMode = null;
  state.shapeCentroid = null;

  initializeAttackers();
  updatePythonPreview();
  updateCentroidDisplay();
  updateStatus("Grid created. Click on the canvas to add vertices.");
  drawGrid();
  applyZoom();
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

  if (state.shapeClosed) {
    updateStatus("Shape is closed. Reset or fill the grid.");
    return;
  }

  const snapped = snapToGrid(x, y);
  const firstVertex = state.vertices[0];

  if (
    !state.shapeClosed &&
    state.vertices.length >= 3 &&
    firstVertex &&
    isSamePoint(snapped, firstVertex)
  ) {
    state.shapeClosed = true;
    drawGrid();
    updateShapeCentroid();
    updateStatus("Shape closed. Click Fill grid to generate the matrix.");
    return;
  }

  const lastVertex = state.vertices[state.vertices.length - 1];
  if (lastVertex && lastVertex.x === snapped.x && lastVertex.y === snapped.y) {
    updateStatus("Vertex already placed there. Choose a different position.");
    return;
  }

  state.vertices.push(snapped);
  state.shapeCentroid = null;
  updateCentroidDisplay();
  updatePythonPreview();
  drawGrid();
  updateStatus(
    state.vertices.length >= 3
      ? "Click Close shape or click the first vertex again to finish."
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
  if (state.shapeClosed) {
    updateStatus("Shape already closed. Fill or reset the grid.");
    return;
  }

  state.shapeClosed = true;
  drawGrid();
  updateShapeCentroid();
  updateStatus("Shape closed. Click Fill grid to generate the matrix.");
}

function resetShape() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }

  state.vertices = [];
  state.shapeClosed = false;
  state.grid = buildGrid(state.rows, state.cols, 1);
  drawGrid();
  updatePythonPreview();
  state.shapeCentroid = null;
  updateCentroidDisplay();
  updateStatus("Shape reset. Start adding vertices again.");
}

function fillGridWithPolygon() {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  if (!state.shapeClosed || state.vertices.length < 3) {
    updateStatus("Close the shape before filling the grid.");
    return;
  }

  const filledGrid = buildGrid(state.rows, state.cols, 1);
  const polygon = state.vertices.slice();
  updateShapeCentroid();

  for (let row = 0; row < state.rows; row += 1) {
    for (let col = 0; col < state.cols; col += 1) {
      const center = {
        x: col * state.cellSize + state.cellSize / 2,
        y: row * state.cellSize + state.cellSize / 2,
      };
      if (isPointInPolygon(center, polygon)) {
        filledGrid[row][col] = 0;
      }
    }
  }

  state.grid = filledGrid;
  drawGrid();
  updatePythonPreview();
  updateStatus("Grid filled. 0 = inside, 1 = outside.");
  downloadJsonPayload(true);
}

function drawGrid() {
  if (!state.hasGrid) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCells();
  drawGridLines();
  drawPolygon();
  drawShapeCentroid();
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

function drawPolygon() {
  if (state.vertices.length === 0) return;

  ctx.strokeStyle = state.shapeClosed ? "#0f7bff" : "#ff7b00";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(state.vertices[0].x, state.vertices[0].y);
  for (let i = 1; i < state.vertices.length; i += 1) {
    ctx.lineTo(state.vertices[i].x, state.vertices[i].y);
  }
  if (state.shapeClosed) {
    ctx.lineTo(state.vertices[0].x, state.vertices[0].y);
  }
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#111827";
  for (const vertex of state.vertices) {
    ctx.beginPath();
    ctx.arc(vertex.x, vertex.y, Math.max(4, state.cellSize * 0.08), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawShapeCentroid() {
  if (!state.shapeCentroid || !state.shapeClosed) return;
  const { pixel } = state.shapeCentroid;
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

function updateShapeCentroid() {
  if (!state.shapeClosed || state.vertices.length < 3) {
    state.shapeCentroid = null;
    updateCentroidDisplay();
    updatePythonPreview();
    return;
  }
  const centroid = computePolygonCentroid(state.vertices);
  if (!centroid) {
    state.shapeCentroid = null;
    updateCentroidDisplay();
    updatePythonPreview();
    return;
  }
  const grid = {
    col: centroid.x / state.cellSize,
    row: centroid.y / state.cellSize,
  };
  state.shapeCentroid = { pixel: centroid, grid };
  updateCentroidDisplay();
  updatePythonPreview();
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

function updateCentroidDisplay() {
  if (!centroidGridEl || !centroidMetersEl) return;
  if (!state.shapeCentroid) {
    centroidGridEl.textContent = "—";
    centroidMetersEl.textContent = "—";
    return;
  }
  const { grid } = state.shapeCentroid;
  const meters = gridToMeters(grid);
  centroidGridEl.textContent = `row ${formatDecimal(grid.row)}, col ${formatDecimal(
    grid.col
  )}`;
  centroidMetersEl.textContent = `${formatDecimal(meters.x)} m, ${formatDecimal(
    meters.y
  )} m`;
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

function getCentroidData() {
  if (!state.shapeCentroid) return null;
  const { grid } = state.shapeCentroid;
  const meters = gridToMeters(grid);
  return {
    grid: {
      row: grid.row,
      col: grid.col,
    },
    meters,
  };
}

function getShapeVerticesData() {
  if (!state.shapeClosed || state.vertices.length < 3) {
    return [];
  }
  return state.vertices.map((vertex) => {
    const grid = {
      col: vertex.x / state.cellSize,
      row: vertex.y / state.cellSize,
    };
    return {
      grid,
      meters: gridToMeters(grid),
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
  updateCentroidDisplay();
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

  const attackerLines = state.attackers
    .map((attacker) => formatAttackerForPython(attacker))
    .join("\n");

  const centroid = getCentroidData();
  const centroidBlock = centroid
    ? [
        "shape_centroid = {",
        `    "grid": {"row": ${formatNumberForPython(
          centroid.grid.row
        )}, "col": ${formatNumberForPython(centroid.grid.col)}},`,
        `    "meters": {"x": ${formatNumberForPython(
          centroid.meters.x
        )}, "y": ${formatNumberForPython(centroid.meters.y)}},`,
        "}",
      ]
    : ["shape_centroid = None"];

  const vertexData = getShapeVerticesData();
  const vertexBlock =
    vertexData.length > 0
      ? [
          "shape_vertices = [",
          ...vertexData.map((vertex) => formatVertexForPython(vertex)),
          "]",
        ]
      : ["shape_vertices = []"];

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

function formatVertexForPython(vertex) {
  const row = formatNumberForPython(vertex.grid.row);
  const col = formatNumberForPython(vertex.grid.col);
  const x = formatNumberForPython(vertex.meters.x);
  const y = formatNumberForPython(vertex.meters.y);
  return `    {"grid": {"row": ${row}, "col": ${col}}, "meters": {"x": ${x}, "y": ${y}}},`;
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

function downloadJsonPayload(isAuto = false) {
  if (!state.hasGrid) {
    updateStatus("Create a grid first.");
    return;
  }
  const payload = buildJsonPayload();
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `grid-${state.rows}x${state.cols}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  if (!isAuto) {
    updateStatus("JSON downloaded.");
  }
}

function buildJsonPayload() {
  const centroid = getCentroidData();
  const vertices = getShapeVerticesData();
  return {
    rows: state.rows,
    cols: state.cols,
    cellSizeMeters: state.cellSizeMeters,
    centroid,
    shapeVertices: vertices,
    matrix: state.grid,
    attackers: state.attackers.map((attacker) => ({
      name: attacker.name,
      start: cellToPayload(attacker.start),
      target: cellToPayload(attacker.target),
    })),
    generatedAt: new Date().toISOString(),
  };
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

