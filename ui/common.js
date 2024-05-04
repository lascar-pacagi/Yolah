const FREE = 3;
const EMPTY = 2;
const BLACK = 0;
const WHITE = 1;
const GRID_DIM = 8;
const BACKGROUND_COLOR = "black";
const RADIUS_FACTOR = 0.5 * 0.7;
const BOARD_DIM_FACTOR = 1;
const INITIAL_BLACK_BB = 0b1000000000000000000000000000100000010000000000000000000000000001n;
const INITIAL_WHITE_BB = 0b0000000100000000000000000001000000001000000000000000000010000000n;
const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const RANKS = ['1', '2', '3', '4', '5', '6', '7', '8'];
let canvas;
let ctx;

function canvasWidth() {
    return Math.round(BOARD_DIM_FACTOR * canvas.width);
}

function canvasHeight() {
    return Math.round(BOARD_DIM_FACTOR * canvas.height);
}

function squareToCoordinates(square) {
    return [Math.floor(square / GRID_DIM), square % GRID_DIM];
}

function possiblePosition(grid, i, j) {
    return i >= 0 && i < GRID_DIM && j >= 0 && j < GRID_DIM && grid[i][j] === FREE;
}

function possibleMoves(grid, square) {
    let res = [];
    let [i, j] = squareToCoordinates(square);
    function line(di, dj) {
        let res = [];
        let ii = i + di;
        let jj = j + dj;
        while (possiblePosition(grid, ii, jj)) {
            res.push([ii, jj]);
            ii += di;
            jj += dj;
        }
        return res;
    }
    for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
            if (di === 0 && dj === 0) continue;
            res = res.concat(line(di, dj));    
        }
    } 
    return res;
}

function atLeastOneMove(grid, bitboard) {
    let square = 0;
    while (bitboard) {
        if (bitboard & 1n) {
            if (possibleMoves(grid, square).length !== 0) {
                return true;
            }
        }
        square++;
        bitboard >>= 1n;
    } 
    return false;
}

function squaresToMove(fromSquare, toSquare) {
    let [i1, j1] = squareToCoordinates(fromSquare);
    let [i2, j2] = squareToCoordinates(toSquare);
    return FILES[j1] + RANKS[i1] + ':' + FILES[j2] + RANKS[i2];
}

function gameOver(grid, blackBb, whiteBb) {
    return !(atLeastOneMove(grid, blackBb) || atLeastOneMove(grid, whiteBb));
}

function bitboardsToGrid(blackBb, whiteBb, emptyBb) {
    let grid = Array(GRID_DIM).fill().map(() => Array(GRID_DIM).fill(FREE));
    for (let i = 0; i < GRID_DIM; i++) {
        for (let j = 0; j < GRID_DIM; j++) {
            const idx = 1n << BigInt(i * GRID_DIM + j);
            if (blackBb & idx) grid[i][j] = BLACK;
            else if (whiteBb & idx) grid[i][j] = WHITE;
            else if (emptyBb & idx) grid[i][j] = EMPTY;
        }
    }
    return grid;
}

function drawStone(x, y, radius, player) {
    ctx.fillStyle = "black";
    if (player === "White") ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();          
    ctx.strokeStyle = "white";
    if (player === "White") ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.stroke();
}

function drawSquareContent(i, j, content, dx, dy) {
    if (content === EMPTY) {
        ctx.fillStyle = BACKGROUND_COLOR;
        ctx.fillRect(j * dx, i * dy, dx, dy);    
    } else if (content === BLACK || content === WHITE) {
        drawStone(j * dx + dx / 2, i * dy + dy / 2, Math.min(dx, dy) * RADIUS_FACTOR, content === BLACK ? "Black" : "White");                 
    }
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function drawPossibleMoves(moves) {
    const dx = canvasWidth() / GRID_DIM;
    const dy = canvasHeight() / GRID_DIM;
    //console.log(dx, dy, canvasWidth(), canvasHeight());
    const marginX = dx * 0.75;
    const marginY = dy * 0.75;
    const w = dx - 2 * marginX;
    const h = dy - 2 * marginY;
    ctx.save();
    ctx.fillStyle = "DarkOrange";
    ctx.strokeStyle = "Black";
    ctx.lineWidth = 1;
    for (const idx in moves) {
        let [i, j] = moves[idx];                
        ctx.fillRect(dx * j + marginX, dy * (GRID_DIM - 1 - i) + marginY, w, h);
        ctx.beginPath();
        ctx.rect(dx * j + marginX, dy * (GRID_DIM - 1 - i) + marginY, w, h);
        ctx.stroke();
    }
    ctx.restore();
}

function drawGrid(grid) {
    const dx = Math.floor(canvasWidth() / GRID_DIM);
    const dy = Math.floor(canvasHeight() / GRID_DIM);
    ctx.fillStyle = BACKGROUND_COLOR;
    ctx.fillRect(0, 0, canvasWidth(), canvasHeight());            
    for (let i = 0; i < GRID_DIM; i++) {                
        for (let j = 0; j < GRID_DIM; j++) {
            if ((i + j) & 1) ctx.fillStyle = "maroon";
            else ctx.fillStyle = "grey";
            ctx.fillRect(j * dx, i * dy, dx, dy);
        }
    }
    for (let i = 0; i < GRID_DIM; i++) {                
        for (let j = 0; j < GRID_DIM; j++) {
            drawSquareContent(GRID_DIM - 1 - i, j, grid[i][j], dx, dy);
        }
    }
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    for (let i = 0; i < GRID_DIM; i++) {                
        drawLine(0, i * dy, canvasWidth(), i * dy);                
    }
    for (let j = 0; j < GRID_DIM; j++) {
        drawLine(j * dx, 0, j * dx, canvasHeight());
    }
}

function clickToSquare(event) {
    const dx = canvasWidth() / GRID_DIM;
    const dy = canvasHeight() / GRID_DIM;
    const x = event.clientX;
    const y = event.clientY;
    return Math.floor(x / dx) + (GRID_DIM - Math.floor(y / dy)) * GRID_DIM;
}

function squareToBitboard(square) {
    return 1n << BigInt(square);
}

function isSet(bitboard, square) {
    return (bitboard & squareToBitboard(square)) !== 0n;
}

function setBitboard(bitboard, square) {
    return bitboard | squareToBitboard(square); 
}

function unsetBitboard(bitboard, square) {
    return bitboard & ~squareToBitboard(square); 
}

function bitboardToSquare(bitboard) {
    let pos = 0;
    while (!(bitboard & 1n)) {
        bitboard = bitboard >> 1n;
        pos++;
    }
    return pos;
}

function ctz(n) {
    let res = 0;
    while (n) {
        if (n & 1n) return res;
        res++;
        n >>= 1n;
    }
    return res;
}

function findMove(oldBb, newBb) {
    const fromSquare = ctz(oldBb & ~newBb); 
    const toSquare   = ctz(newBb & ~oldBb);
    return squaresToMove(Number(fromSquare), Number(toSquare));
}

function lastMove(oldBlackBb, oldWhiteBb, newBlackBb, newWhiteBb) {
    if (oldBlackBb === newBlackBb && oldWhiteBb === newWhiteBb) {
        return "";
    }
    if (oldBlackBb === newBlackBb) {
        return `[white move] ${findMove(oldWhiteBb, newWhiteBb)}`;
    }
    return `[black move] ${findMove(oldBlackBb, newBlackBb)}`;    
}

function doSleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms)); 
}

async function sleep(ms) {
    await doSleep(ms);
}

function drawArrow(fromx, fromy, tox, toy, arrowWidth, color){
    const headlen = 10;
    const angle = Math.atan2(toy-fromy,tox-fromx);
 
    ctx.save();
    ctx.strokeStyle = color;
 
    //starting path of the arrow from the start square to the end square
    //and drawing the stroke
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.lineWidth = arrowWidth;
    ctx.stroke();
 
    //starting a new path from the head of the arrow to one of the sides of
    //the point
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),
               toy-headlen*Math.sin(angle-Math.PI/7));
 
    //path from the side point of the arrow, to the other side point
    ctx.lineTo(tox-headlen*Math.cos(angle+Math.PI/7),
               toy-headlen*Math.sin(angle+Math.PI/7));
 
    //path from the side point back to the tip of the arrow, and then
    //again to the opposite side point
    ctx.lineTo(tox, toy);
    ctx.lineTo(tox-headlen*Math.cos(angle-Math.PI/7),
               toy-headlen*Math.sin(angle-Math.PI/7));
 
    ctx.stroke();
    ctx.restore();
}

function squareToXY([i, j]) {
    const dx = canvasWidth() / GRID_DIM;
    const dy = canvasHeight() / GRID_DIM;
    const x = j * dy + dy / 2;
    const y = (GRID_DIM - i - 1) * dx + dx / 2;
    return [x, y];
}

function drawArrowBetweenSquares([[i1, j1], [i2, j2]], arrowWidth, color) {
    const [x1, y1] = squareToXY([i1, j1]);
    const [x2, y2] = squareToXY([i2, j2]);
    drawArrow(x1, y1, x2, y2, arrowWidth, color);
}

function lastMoveCoordinates(oldBlackBb, oldWhiteBb, newBlackBb, newWhiteBb) {
    function findMove(oldBb, newBb) {
        const fromSquare = ctz(oldBb & ~newBb); 
        const toSquare   = ctz(newBb & ~oldBb);
        return [squareToCoordinates(fromSquare), squareToCoordinates(toSquare)];
    }
    if (oldBlackBb === newBlackBb && oldWhiteBb === newWhiteBb) {
        return [[0, 0], [0, 0]];
    }
    if (oldBlackBb === newBlackBb) {
        return findMove(oldWhiteBb, newWhiteBb);
    }
    return findMove(oldBlackBb, newBlackBb);
}
