function Grid(size, previousState, value_jpg_refs) {
  this.isMaravillas = 1;
  this.size = size;
  this.cells = previousState ? this.fromState(previousState) : this.empty();
  if (value_jpg_refs) {
      this.value_jpg_refs = value_jpg_refs
  }
  else {
      this.value_jpg_refs = {}
      this.value_jpg_refs[-11] = 978;
      this.value_jpg_refs[-12] = 979;
      this.value_jpg_refs[-1] = 988;
      this.value_jpg_refs[-2] = 989;
      this.value_jpg_refs[2] = 0 + Math.floor(Math.random() * 4);
      this.value_jpg_refs[4] = 4 + Math.floor(Math.random() * 3);
      this.value_jpg_refs[8] = 8 + Math.floor(Math.random() * 3);
      this.value_jpg_refs[16] = 16 + Math.floor(Math.random() * 3);
      this.value_jpg_refs[32] = 32 + Math.floor(Math.random() * 2);
      this.value_jpg_refs[64] = 64 + Math.floor(Math.random() * 4);
      this.value_jpg_refs[128] = 128 + Math.floor(Math.random() * 2);
      this.value_jpg_refs[256] = 256 + Math.floor(Math.random() * 4);
      this.value_jpg_refs[512] = 512 + Math.floor(Math.random() * 5);
      this.value_jpg_refs[1024] = 1024 + Math.floor(Math.random() * 2);
      this.value_jpg_refs[2048] = 2048;
      this.value_jpg_refs[4096] = 4096;
  }
}

// Build a grid of the specified size
Grid.prototype.empty = function () {
  var cells = [];

  for (var x = 0; x < this.size; x++) {
    var row = cells[x] = [];

    for (var y = 0; y < this.size; y++) {
      row.push(null);
    }
  }

  return cells;
};

Grid.prototype.fromState = function (state) {
  var cells = [];

  for (var x = 0; x < this.size; x++) {
    var row = cells[x] = [];

    for (var y = 0; y < this.size; y++) {
      var tile = state[x][y];
      row.push(tile ? new Tile(tile.position, tile.value) : null);
    }
  }

  return cells;
};

// Find the first available random position
Grid.prototype.randomAvailableCell = function () {
  var cells = this.availableCells();

  if (cells.length) {
    return cells[Math.floor(Math.random() * cells.length)];
  }
};

Grid.prototype.availableCells = function () {
  var cells = [];

  this.eachCell(function (x, y, tile) {
    if (!tile) {
      cells.push({ x: x, y: y });
    }
  });

  return cells;
};

// Call callback for every cell
Grid.prototype.eachCell = function (callback) {
  for (var x = 0; x < this.size; x++) {
    for (var y = 0; y < this.size; y++) {
      callback(x, y, this.cells[x][y]);
    }
  }
};

// Check if there are any cells available
Grid.prototype.cellsAvailable = function () {
  return !!this.availableCells().length;
};

// Check if the specified cell is taken
Grid.prototype.cellAvailable = function (cell) {
  return !this.cellOccupied(cell);
};

Grid.prototype.cellOccupied = function (cell) {
  return !!this.cellContent(cell);
};

Grid.prototype.cellContent = function (cell) {
  if (this.withinBounds(cell)) {
    return this.cells[cell.x][cell.y];
  } else {
    return null;
  }
};

// Inserts a tile at its position
Grid.prototype.insertTile = function (tile) {
  this.cells[tile.x][tile.y] = tile;
};

Grid.prototype.removeTile = function (tile) {
  this.cells[tile.x][tile.y] = null;
};

Grid.prototype.withinBounds = function (position) {
  return position.x >= 0 && position.x < this.size &&
         position.y >= 0 && position.y < this.size;
};

Grid.prototype.serialize = function () {
  var cellState = [];

  for (var x = 0; x < this.size; x++) {
    var row = cellState[x] = [];

    for (var y = 0; y < this.size; y++) {
      row.push(this.cells[x][y] ? this.cells[x][y].serialize() : null);
    }
  }

  return {
    size: this.size,
    cells: cellState,
    value_jpg_refs: this.value_jpg_refs
  };
};
