# snake.asm — Snake for demoniVM-2, M1 assembly
#
# mem layout:
#   hx, hy       — head position
#   dir          — 0=right 1=down 2=left 3=up
#   fx, fy       — food position
#   ftog         — food toggle (0=pos A, 1=pos B)
#   key          — last key code (from INPUT)
#   tx, ty       — tail[0] position (one segment behind head)
#
# Controls: wasd  (w=up s=down a=left d=right)
# Eat food → food moves to alternating fixed position
# Hit wall → restart (JMP 0)
# Quit    → 'q' handled by VSYNC in host

# ── init ──────────────────────────────────────────────────────────────────
PUSH 5   STORE hx      # head starts at (5, 10)
PUSH 10  STORE hy
PUSH 0   STORE dir     # moving right
PUSH 8   STORE fx      # food A at (8, 5)
PUSH 5   STORE fy
PUSH 0   STORE ftog
PUSH 5   STORE tx      # tail starts at head pos (hidden under it)
PUSH 10  STORE ty

# ── main loop ─────────────────────────────────────────────────────────────
loop:
  CLS 0

  # poll input → store key
  INPUT  STORE key

  # w = 119 → dir = 3 (up)
  LOAD key  PUSH 119  EQ  JZ skip_w
  PUSH 3  STORE dir
skip_w:
  # s = 115 → dir = 1 (down)
  LOAD key  PUSH 115  EQ  JZ skip_s
  PUSH 1  STORE dir
skip_s:
  # a = 97 → dir = 2 (left)
  LOAD key  PUSH 97   EQ  JZ skip_a
  PUSH 2  STORE dir
skip_a:
  # d = 100 → dir = 0 (right)
  LOAD key  PUSH 100  EQ  JZ skip_d
  PUSH 0  STORE dir
skip_d:

  # shift tail: tail ← head
  LOAD hx  STORE tx
  LOAD hy  STORE ty

  # move head based on dir
  LOAD dir  PUSH 0  EQ  JZ skip_right
  LOAD hx  PUSH 1  ADD  STORE hx
skip_right:
  LOAD dir  PUSH 1  EQ  JZ skip_down
  LOAD hy  PUSH 1  ADD  STORE hy
skip_down:
  LOAD dir  PUSH 2  EQ  JZ skip_left
  LOAD hx  PUSH 1  SUB  STORE hx
skip_left:
  LOAD dir  PUSH 3  EQ  JZ skip_up
  LOAD hy  PUSH 1  SUB  STORE hy
skip_up:

  # wall collision → restart
  LOAD hx  PUSH 31  GT  JZ wall_ok_1
  JMP 0
wall_ok_1:
  LOAD hx  PUSH 0   LT  JZ wall_ok_2
  JMP 0
wall_ok_2:
  LOAD hy  PUSH 19  GT  JZ wall_ok_3
  JMP 0
wall_ok_3:
  LOAD hy  PUSH 0   LT  JZ wall_ok_4
  JMP 0
wall_ok_4:

  # food collision: if hx==fx AND hy==fy → eat
  LOAD hx  LOAD fx  EQ
  LOAD hy  LOAD fy  EQ
  MUL
  JZ no_food
  # toggle food position A (8,5) ↔ B (22,14)
  LOAD ftog  JZ use_food_a
  PUSH 8   STORE fx
  PUSH 5   STORE fy
  PUSH 0   STORE ftog
  JMP food_done
use_food_a:
  PUSH 22  STORE fx
  PUSH 14  STORE fy
  PUSH 1   STORE ftog
food_done:
no_food:

  # draw tail (color 2)
  LOAD tx  LOAD ty  PUSH 2  PSET

  # draw head (color 3)
  LOAD hx  LOAD hy  PUSH 3  PSET

  # draw food (color 1 = dot)
  LOAD fx  LOAD fy  PUSH 1  PSET

  VSYNC
  JMP loop
