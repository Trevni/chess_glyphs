// Minimal canvas-based chess board with drag moves and server-side legality.
// No external libraries required.

const board = document.getElementById('board');
const overlay = document.getElementById('overlay');
const ctxB = board.getContext('2d');
const ctxO = overlay.getContext('2d');

const fenInput = document.getElementById('fen-input');
const btnSetFen = document.getElementById('btn-setfen');
const btnCopyFen = document.getElementById('btn-copyfen');
const btnStart = document.getElementById('btn-startpos');
const btnUndo = document.getElementById('btn-undo');
const btnRedo = document.getElementById('btn-redo');

const chkAttacks = document.getElementById('chk-attacks');
const chkHanging = document.getElementById('chk-hanging');
const chkPins = document.getElementById('chk-pins');
const chkKing = document.getElementById('chk-king');
const chkRulebased = (()=>{const l=document.createElement('label'); l.innerHTML='<input type="checkbox" id="chk-rule"> Rule-based only'; document.querySelector('.controls .row.small').appendChild(l); return document.getElementById('chk-rule');})();

// Position state
let fen = "startpos";
let history = [];
let future = [];

function setFEN(newFen){
  fen = newFen;
  fenInput.value = fen;
  drawBoard();
  refreshGlyphs();
}

function startPositionFEN(){
  return "rnbqkbnr/pppppppp/8/8/8/8/pppppppp/rnbqkbnr w kqKQ - 0 1".replace('pppppppp/rnbqkbnr','PPPPPPPP/RNBQKBNR'); // ensure white on bottom
}
// Piece map using Unicode
const PIECES = {
  'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
  'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'
};

function parseFENToArray(fenStr){
  // returns [8][8] chars or null
  const fields = fenStr.split(' ');
  const rows = fields[0].split('/');
  const boardArr = [];
  for (let r=0;r<8;r++){
    const row = [];
    for (const ch of rows[r]){
      if (/\d/.test(ch)){
        for (let i=0;i<parseInt(ch);i++) row.push(null);
      } else {
        row.push(ch);
      }
    }
    boardArr.push(row);
  }
  return boardArr; // ranks 8..1 as rows 0..7
}

function drawBoard(){
  const W = board.width, H = board.height;
  const S = W/8;
  ctxB.clearRect(0,0,W,H);
  // squares
  for (let r=0;r<8;r++){
    for (let c=0;c<8;c++){
      ctxB.fillStyle = ((r+c)%2==0) ? '#f0d9b5' : '#b58863';
      ctxB.fillRect(c*S, r*S, S, S);
    }
  }
  // pieces
  const arr = parseFENToArray(fen);
  ctxB.font = `${Math.floor(S*0.8)}px serif`;
  ctxB.textAlign = 'center';
  ctxB.textBaseline = 'middle';
  for (let r=0;r<8;r++){
    for (let c=0;c<8;c++){
      const p = arr[r][c];
      if (!p) continue;
      ctxB.fillStyle = (p === p.toUpperCase()) ? '#111' : '#111';
      ctxB.fillText(PIECES[p], c*S+S/2, r*S+S/2+2);
    }
  }
}

function drawOverlay(glyphs){
  const W = overlay.width, H = overlay.height;
  const S = W/8;
  ctxO.clearRect(0,0,W,H);

  const drawHeat = (grid, rgba) => {
    let max = 0;
    for (let r=0;r<8;r++) for (let c=0;c<8;c++) max = Math.max(max, grid[r][c]);
    if (max <= 0) return;
    for (let r=0;r<8;r++){
      for (let c=0;c<8;c++){
        const v = grid[r][c];
        if (v <= 0) continue;
        const a = Math.min(1, v / max) * rgba[3];
        ctxO.fillStyle = `rgba(${rgba[0]},${rgba[1]},${rgba[2]},${a})`;
        ctxO.fillRect(c*S, r*S, S, S);
      }
    }
  };
  const drawFlags = (grid, color, marker='•') => {
    ctxO.fillStyle = color;
    ctxO.font = `${Math.floor(S*0.5)}px sans-serif`;
    ctxO.textAlign = 'center';
    ctxO.textBaseline = 'middle';
    for (let r=0;r<8;r++){
      for (let c=0;c<8;c++){
        if (grid[r][c] > 0.5){
          ctxO.fillText(marker, c*S + S/2, r*S + S/2);
        }
      }
    }
  };

  if (chkAttacks.checked){
    drawHeat(glyphs.attack_white, [255,0,0,0.35]);
    drawHeat(glyphs.attack_black, [0,0,255,0.35]);
  }
  if (chkHanging.checked){
    drawFlags(glyphs.hanging_white, 'rgba(255,165,0,0.9)', 'H');
    drawFlags(glyphs.hanging_black, 'rgba(255,165,0,0.9)', 'H');
  }
  if (chkPins.checked){
    drawFlags(glyphs.pinned_white, 'rgba(128,0,128,0.9)', 'P');
    drawFlags(glyphs.pinned_black, 'rgba(128,0,128,0.9)', 'P');
  }
  if (chkKing.checked){
    drawHeat(glyphs.king_danger_white, [255,0,0,0.25]);
    drawHeat(glyphs.king_danger_black, [255,0,0,0.25]);
  }
}

async function refreshGlyphs(){
  const res = await fetch(`/predict?fen=${encodeURIComponent(fen)}${(typeof chkRulebased!=='undefined' && chkRulebased.checked)?'&rb=1':''}`);
  const data = await res.json();
  if (data.glyphs){
    drawOverlay(data.glyphs);
  }
}

// Drag handling
let dragFrom = null;
board.addEventListener('mousedown', (e) => {
  const rect = board.getBoundingClientRect();
  const S = rect.width / 8;
  const c = Math.floor((e.clientX - rect.left)/S);
  const r = Math.floor((e.clientY - rect.top)/S);
  dragFrom = {r, c};
});

board.addEventListener('mouseup', async (e) => {
  if (!dragFrom) return;
  const rect = board.getBoundingClientRect();
  const S = rect.width / 8;
  const c = Math.floor((e.clientX - rect.left)/S);
  const r = Math.floor((e.clientY - rect.top)/S);
  const from = rcToSquare(dragFrom.r, dragFrom.c);
  const to = rcToSquare(r, c);
  dragFrom = null;

  // Construct UCI (auto-queen on promotion if needed, naive)
  let uci = from + to;
  // simple promotion guess if moving to last rank
  if (uci.match(/^[a-h][27][a-h][18]$/)) uci += 'q';

  const res = await fetch('/move', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({fen, uci})
  });
  const data = await res.json();
  if (data.error){
    // illegal; do nothing
    return;
  }
  if (data.fen){
    history.push({fen});
    future = [];
    setFEN(data.fen);
    if (data.glyphs) drawOverlay(data.glyphs);
  }
});

function rcToSquare(r, c){
  const file = 'abcdefgh'[c];
  const rank = (8 - r).toString();
  return file + rank;
}

function resize(){
  const wrap = document.getElementById('board-wrap');
  const w = wrap.clientWidth;
  board.width = w; board.height = w;
  overlay.width = w; overlay.height = w;
  drawBoard();
  refreshGlyphs();
}
window.addEventListener('resize', resize);

// Controls
btnStart.onclick = () => {
  history = []; future = [];
  setFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
};
btnUndo.onclick = () => {
  if (history.length){
    const last = history.pop();
    future.push({fen});
    setFEN(last.fen);
  }
};
btnRedo.onclick = () => {
  if (future.length){
    const nxt = future.pop();
    history.push({fen});
    setFEN(nxt.fen);
  }
};
btnSetFen.onclick = () => {
  const f = fenInput.value.trim();
  if (!f) return;
  try{
    setFEN(f);
  }catch(e){
    alert('Invalid FEN');
  }
};
btnCopyFen.onclick = async () => {
  await navigator.clipboard.writeText(fen);
};

[chkAttacks, chkHanging, chkPins, chkKing].forEach(el => el.addEventListener('change', refreshGlyphs));

// Init
window.addEventListener('load', () => {
  setFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  resize();
});
