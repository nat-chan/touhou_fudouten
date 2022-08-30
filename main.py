import pygame as pg; from pygame.locals import *
import numpy as np
from addict import Dict as addict_dict
import colorsys
from typing import *
import sys
import math
import argparse
from random import random
from functools import lru_cache
#238, 102

class Dict(addict_dict):
    def at(self, k: int) -> Any:
        return self[list(self.keys())[k]]

LEVEL = [4, 7, 10]
CONFIG = Dict()
BPM = 142
DISPLAY_RECT = pg.Rect(0, 0, 1280, 960)
BORDER_RECT = pg.Rect(62, 30, 834-62, 930-30)
PLAYAREA_RECT = pg.Rect(0, 0, 2*386, 2*450)
DISPLAY_CENTER = np.array([DISPLAY_RECT.left*.5+DISPLAY_RECT.right*.5, DISPLAY_RECT.top*.5+DISPLAY_RECT.bottom*.5])
PLAYAREA_CENTER = np.array([PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.top*.5+PLAYAREA_RECT.bottom*.5])
PLAYAREA_MARGIN = 40
DEFAULT_POS = np.array((PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.bottom+PLAYAREA_MARGIN))
TELEPORT = 120
INFO_RECT = pg.Rect(834+30, 30, 1280-834-2*30, 500)
mpf = round(1000/60) # 16.6666 ~ 17 t->msへの変換に使う
beat_t = Tuple[int, int, int, int]

cl = Dict(pg.colordict.THECOLORS) # https://www.pygame.org/docs/ref/color_list.html
pr = Dict(
    はっぱや=np.array([0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.65, 0.0], dtype=np.float64),
    さんかく=np.array([0.0, 0.0, 0.45, 0.5, 0.0, 0.0, 0.45, -0.5], dtype=np.float64),
    くるくる=np.array([0.4614, 0.4614, 0, 0, 0.622, 0.196, 0, 0], dtype=np.float64),
    たちきの=np.array([0, 0, 0.3, 0.3, 0, 0, 41 / 50, 0], dtype=np.float64),
    ひしがた=np.array([0, 0, 0.5, 0.5, 0, 0, -0.5, 0.5], dtype=np.float64),
    かいそう=np.array([0.4614, 0.4614, 0, 0, 0, 0, 0.2896, 0.585], dtype=np.float64),
    かこまれ=np.array([0, 0, -0.5, -0.5, 0, 0, -0.5, 0.5], dtype=np.float64),
    せきへん=np.array([0,0,0.5,0.25,0, 0, 0.5, -0.25], dtype=np.float64),
    てふてふ=np.array([0,-0.25,-0.5,0.75,0, 0.25, 0.5, 0], dtype=np.float64),
    みつびし=np.array([-0.25,0.5,0,0,0,0, 0.75, 0], dtype=np.float64),
    ひびわれ=np.array([0.4614, 0.4614, 0, 0, 0, 0, 0.2896, -0.585], dtype=np.float64),
    くろすい=np.array([0, 0.7071, 0, 0, 0.5, 0, 0, 0], dtype=np.float64),
    スペード=np.array([0, 0.75, 0.25, 0, 0, 0.75, 0.25, 0], dtype=np.float64),
    うちゅう=np.array([2/3, -0.5, 0, 0, -0.25, 0.5, 0, 0], dtype=np.float64),
    ドラゴン=np.array([0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25], dtype=np.float64),
    おてがみ=np.array([0, 0, 0, 0.7071, 0, 0, -0.5, 0], dtype=np.float64),
    はじっこ=np.array([0, 0, 0, -0.5, 0.8, 0, 0, 0], dtype=np.float64),
)

class se: #se用の名前空間
    @classmethod
    def load(se):
        se.cancel00 = pg.mixer.Sound("data/se_cancel00.wav")
        se.ok00 = pg.mixer.Sound("data/se_ok00.wav")
        se.slash = pg.mixer.Sound("data/se_slash.wav")
        se.pldead00 = pg.mixer.Sound("data/se_pldead00.wav")
        se.invalid = pg.mixer.Sound("data/se_invalid.wav")
        se.extend = pg.mixer.Sound("data/se_extend.wav")

class sp: #sprite用の名前空間
    @classmethod
    def load(sp):
        sp.bg = pg.image.load("data/bg.png")
        sp.logo = pg.image.load("data/logo.png")
        sp.description = pg.image.load("data/description5.png")
        sp.pl00 = pg.image.load("data/w2x_pl00.png")
        sp.pl10 = pg.image.load("data/w2x_pl10.png")
        sp.sloweffect = pg.Surface((128, 128), pg.SRCALPHA)
        sp.sloweffect.blit(pg.image.load("data/w2x_eff_sloweffect.png"), (0, 0), (0,0, 128, 128))
        sp.eff_charge = pg.Surface((64, 64), pg.SRCALPHA)
        sp.eff_charge.blit(pg.image.load("data/w2x_eff_charge.png"), (0, 0), (0,0, 128, 128))
        sp.etama7a = pg.image.load("data/etama7a.png")
        sp.houi2 = pg.image.load("data/houi2.png")
        sp.witch1 = pg.image.load("data/witch1.png")
        sp.yukari_bs = pg.image.load("data/yukari_bs.png")
        sp.laspetxt = pg.image.load("data/laspetxt.png")
        sp.title_bg = pg.image.load("data/title_bg.png")
        sp.reimu_l = pg.image.load("data/reimu_l.png")
        sp.marisa_r = pg.image.load("data/marisa_r.png")
        sp.yukari_r = pg.image.load("data/yukari_r.png")

@lru_cache(maxsize=None)
def font(size, name="data/x12y16pxMaruMonica.ttf"):
    if name[:5] == "data/":
        return  pg.font.Font(name, size)
    return pg.font.SysFont(name, size)

def lifegame_step(a):
    DH = (-1,-1,-1,0,0,1,1,1)
    DW = (-1,0,1,-1,1,-1,0,1)
    H = len(a); W = len(a[0])
    b = [[False]*W for _ in range(H)]
    for h in range(1, H-1):
        for w in range(1, W-1):
            s = sum(a[h+dh][w+dw] for dh, dw in zip(DH, DW))
            if a[h][w] and 1 < s < 4 or not a[h][w] and s == 3:
                b[h][w] = True
    return b

def lifegame_str2mat(lines, pad=1):
    W = len(lines[0])
    out = [[False]*(W+2*pad) for _ in range(pad)]
    out += [[False]*pad+[l == "X" for l in line]+[False]*pad for line in lines]
    out += [[False]*(W+2*pad) for _ in range(pad)]
    return out
def lifegame_mat2str(lines):
    return ["".join("X" if l else "." for l in line) for line in lines]

GALAXY = [lifegame_str2mat([
"XXXXXX.XX",
"XXXXXX.XX",
".......XX",
"XX.....XX",
"XX.....XX",
"XX.....XX",
"XX.......",
"XX.XXXXXX",
"XX.XXXXXX"],pad=3)]
for i in range(7): GALAXY.append(lifegame_step(GALAXY[i]))

################ ?HE 補助関数群 ################
def check_good_ending():
    return CONFIG["bomb_stock"] < LEVEL[2] and CONFIG["ms"] == 0
def display_l(screen, sprite):
    w, h = sprite.get_size()
    screen.blit(sprite, (0, DISPLAY_RECT.height-h))
def display_r(screen, sprite):
    w, h = sprite.get_size()
    screen.blit(sprite, (DISPLAY_RECT.width-w, DISPLAY_RECT.height-h))


def hatafast(p: np.ndarray, rep=2) -> np.ndarray:
    """畑写像を高速に計算"""
    z = np.array([[1, 1, 1]], dtype=np.float64)
    A0 = [
        np.array([[p[0] + p[2], p[1] + p[3], 0],
                  [-p[1] + p[3], p[0] - p[2], 0], [0, 0, 1]], dtype=np.float64),
        np.array(
            [[p[4] + p[6], p[5] + p[7], 0], [-p[5] + p[7], p[4] - p[6], 0],
             [-p[4] - p[6] + 1, -p[5] - p[7], 1]], dtype=np.float64)
    ]
    A1 = [A0[i & 1] @ A0[(i >> 1) & 1] for i in range(4)]
    A = np.hstack([A1[i & 3] @ A1[(i >> 2) & 3] for i in range(16)])
    for _ in range(rep):
        z = (z @ A).reshape(-1, 3)
    return z[:, :-1]

def z2xy(z: np.ndarray) -> Tuple[int, int]:
    """画面に収まる位置に座標をスケール"""
    scale = min(PLAYAREA_RECT.width, PLAYAREA_RECT.height)/2
    x = z[0]*scale+PLAYAREA_RECT.width/2
    y = -z[1]*scale+PLAYAREA_RECT.height/2
    return (int(x), int(y))

def hsv2rgb(h: float,s: float,v: float) -> Tuple[int, int, int]:
    return tuple(round(i*255) for i in colorsys.hsv_to_rgb(h,s,v))


def circ(center: np.ndarray, angle: float, radius: float) -> np.ndarray:
    """円周上の与えられた角度の位置の座標を返す"""
    return np.array(center)+radius*np.array([np.cos(angle), np.sin(angle)])

def calc_d(angle: float) -> float:
    """32x32等の円形スプライトを回転させる時の中心からのズレ"""
    return math.cos(angle%90/360*2*np.pi)+math.sin(angle%90/360*2*np.pi)

# beat(拍数)関連の補助関数
def beat2count(beat: beat_t) -> int:
    return beat[0]+4*beat[1]+16*beat[2]+64*beat[3]
def count2beat(count: int) -> beat_t:
    return (count%4, count//4%4, count//16%4, count//64)
def ms2beat(ms: int) -> beat_t:
    count = int(BPM/(60*1000)*ms)
    beat = count2beat(count)
    assert beat[0]+4*beat[1]+16*beat[2]+64*beat[3] == count
    return beat
def beat2ms(beat: beat_t) -> int:
    count = beat2count(beat)
    ms = count/(BPM/(60*1000))
    return ms
def beat2squares(beat: beat_t) -> List[str]:
    if 8 < beat[3]: return ["■"*4 for _ in range(5)]
    squares = [["□"]*4 for _ in range(5)]
    for i in range(4):
        squares[i][beat[i]%4] = "■"
    squares[4][beat[i]//4%4] = "■"
    return reversed(["".join(s) for s in squares])

def square_dist(a, b):
    """aとbの距離の2乗"""
    return (a.pos[0]-b.pos[0])**2 + (a.pos[1]-b.pos[1])**2

def check_hit(a, b) -> bool:
    """被弾判定、東方は円どうしの交点が直交するまで重なるが特に考慮せずに通常の判定"""
    return square_dist(a, b) < (a.radius+b.radius)**2

def r() -> float: return 2*np.pi*random()

################ ?MI メインクラス群 ################
class Status:
    """時機の現在の状態"""
    KEY = ["left", "right", "up", "down", "slow", "shot", "bomb"]
    def __init__(self) -> None:
        self.now = False
        self.last = [-float("inf"), -float("inf")] # 負の無限時間から定常状態

class Beats(list):
    """現在の拍子を管理するクラス、長さ2のリストと同様"""
    @staticmethod
    def wildcard_equal(self0: beat_t, other: beat_t):
        return all([s == o or o is None for s, o in zip(self0, other)])
    def ignite(self, *beat: beat_t) -> bool:
        """現フレームで指定した拍子に到達したときに発火する"""
        return self[0] != self[1] and self.wildcard_equal(self[0], beat)

class MyController: # joystick対応を視野に入れてclass化
    def check_input(self) -> Dict:
        """ユーザの現在の入力を返す"""
        keys = pg.key.get_pressed()
        return Dict(
            left =keys[pg.K_LEFT],
            right=keys[pg.K_RIGHT],
            up   =keys[pg.K_UP],
            down =keys[pg.K_DOWN],
            slow =keys[pg.K_LSHIFT],
            bomb =keys[pg.K_x],
            shot =keys[pg.K_z],
            esc  =keys[pg.K_ESCAPE],
            retry=keys[pg.K_r],
            item =keys[pg.K_c],
        )

class Interpolation:
    def __init__(self, start_beat, end_beat, start_pos, end_pos):
            self.start_ms = beat2ms(start_beat)
            self.end_ms = beat2ms(end_beat)
            self.start_pos = start_pos
            self.end_pos = end_pos
    def __call__(self, ms):
            p = (ms-self.start_ms)/(self.end_ms-self.start_ms)
            return (1-p)*self.start_pos+p*self.end_pos

class Dialog(pg.Surface):
    def __init__(self, rect, fheight=30, margin=20, border=2, border_color=cl.white) -> None:
        super().__init__((rect.width, rect.height))
        self.rect = rect
        self.font = font(fheight)
        self.margin = margin
        self.offset = margin
        self.border = border
        self.border_color = border_color
    def print(self, txts: str = "", color=cl.white) -> None:
        for txt in txts.split("\n"):
            self.blit(self.font.render(txt, True, color), (self.margin, self.offset))
            self.offset += self.font.get_height()
    def draw_border(self) -> None:
        pg.draw.rect(self, self.border_color, self.rect, self.border)
    def start(self, bg_color) -> None:
        self.offset = self.margin
        self.fill(bg_color)
    def end(self, screen: pg.Surface) -> None:
        screen.blit(self, (self.rect.left, self.rect.top))
        pg.draw.rect(screen, self.border_color, self.rect, self.border)

################ ?RE 霊夢(自機) ################
class Reimu:
    """自機のクラス"""
    def __init__(self) -> None:
        self.pos = DEFAULT_POS.copy()
        self.option = pg.Surface((32, 32), pg.SRCALPHA)
        self.option.blit(sp.pl00, (0, 0), (3*32*2, 3*48*2, 16*2, 16*2))
        self.bomb_stock = CONFIG["bomb_stock"]
        self.bomb_invincible = False
        self.bomb_invincible_time = 2*1000 # ボム無敵時間
        self.bomb_lasttime = float("inf")
        self.hit_invincible = False
        self.hit_invincible_time = 1*1000 # 被弾無敵時間
        self.spellcard_invincible = False
        self.spellcard_invincible_time = 0.5*1000 # スペルカード移行時無敵
        self.hit_lasttime = float("inf")
        self.radius = 5 # 通常は3.0, 2.4の2倍
        self.kurai_bomb = 30*mpf # 通常は8、15でもよさそう ここはフレーム指定
        self.status = Dict({k:Status() for k in Status.KEY})
        self.controllable = True

    def update(self, t: int, ms: int, controller_input: Dict, is_hit: bool) -> None:
        """自機の位置、無敵等をupdate"""
        # controller inputとreimu.statusが変化したら更新し、更新時刻を記録する
        for k in Status.KEY:
            if self.status[k].now^controller_input[k]:
                self.status[k].last[controller_input[k]] = ms
                self.status[k].now = controller_input[k]

        speed = 2*[4.0, 2.0][controller_input.slow]
        if [controller_input.left, controller_input.right, controller_input.up, controller_input.down].count(True) == 2:
            speed /= math.sqrt(2)

        if self.controllable:
            if controller_input.left:    self.pos[0] -= speed
            elif controller_input.right: self.pos[0] += speed
            if controller_input.up:      self.pos[1] -= speed
            elif controller_input.down:  self.pos[1] += speed
        
        # 画面外脱出防止
        if self.pos[0] < PLAYAREA_RECT.left+PLAYAREA_MARGIN:   self.pos[0] = PLAYAREA_RECT.left+PLAYAREA_MARGIN
        if self.pos[0] > PLAYAREA_RECT.right-PLAYAREA_MARGIN:  self.pos[0] = PLAYAREA_RECT.right-PLAYAREA_MARGIN
        if self.pos[1] < PLAYAREA_RECT.top+PLAYAREA_MARGIN:    self.pos[1] = PLAYAREA_RECT.top+PLAYAREA_MARGIN
        if self.pos[1] > PLAYAREA_RECT.bottom-PLAYAREA_MARGIN: self.pos[1] = PLAYAREA_RECT.bottom-PLAYAREA_MARGIN

        if self.controllable:
            # 通常のボムの処理
            if ( self.status["bomb"].now and 0 < self.bomb_stock 
                and not self.bomb_invincible and not self.hit_invincible ):
                self.bomb_invincible = True
                self.bomb_stock -= 1
                self.bomb_lasttime = ms
                se.slash.play()

            # 通常の被弾の処理
            elif is_hit and not self.bomb_invincible and not self.hit_invincible and not self.spellcard_invincible:
                self.hit_invincible = True
                self.bomb_stock -= 2
                self.hit_lasttime = ms
                se.pldead00.play()

            # 喰らいボムの処理
            elif ( self.status["bomb"].now and 0 < self.bomb_stock and self.hit_invincible ): 
                if ms-self.hit_lasttime <= self.kurai_bomb:
                    self.hit_invincible = False
                    self.bomb_invincible = True
                    self.bomb_stock += 1
                    self.bomb_lasttime = ms
                    se.slash.play()

            # ボム不可時の処理
            elif ( self.status["bomb"].now and 0 >= self.bomb_stock 
                and not self.bomb_invincible and not self.hit_invincible ):
                se.invalid.play()

        # 無敵時間終了処理
        if 0 < ms-self.bomb_lasttime-self.bomb_invincible_time:
            self.bomb_invincible = False
        if 0 < ms-self.hit_lasttime-self.hit_invincible_time:
            self.hit_invincible = False
        if 0 < ms-self.hit_lasttime-self.hit_invincible_time:
            self.hit_invincible = False
        if 0 < ms-self.spellcard_lasttime-self.spellcard_invincible_time:
            self.spellcard_invincible = False
    
    def guard_spellcard(self, t: int, ms: int) -> None:
        self.spellcard_invincible = True
        self.spellcard_lasttime = ms

    def draw_upper(self, t: int, ms: int, screen: pg.Surface) -> None:
        """Surfaceへの自機の描画 (弾より上のレイヤー)"""
        angle_sloweffect = (ms*18/100)%360
        d2 = 2*calc_d(angle_sloweffect)
        rot_sloweffect = pg.transform.rotate(sp.sloweffect, angle_sloweffect)
        rot_sloweffect2 = pg.transform.rotate(sp.sloweffect, -angle_sloweffect)
        if self.status.slow.now:
            screen.blit(rot_sloweffect, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))
            screen.blit(rot_sloweffect2, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))

    def draw_lower(self, t: int, ms: int, screen: pg.Surface) -> None:
        """Surfaceへの自機の描画 (弾より下のレイヤー)"""
        if self.status.left.now: reimu_offset = 1
        elif self.status.right.now: reimu_offset = 2
        else: reimu_offset = 0

        screen.blit(
            sp.pl10 if any([self.bomb_invincible, self.hit_invincible, self.spellcard_invincible]) else sp.pl00,
            self.pos-(16*2, 24*2), (0+32*((ms//mpf)//8%4)*2, reimu_offset*48*2, 32*2, 48*2))
        if self.bomb_invincible:
            bomb_radius = self.bomb_invincible_time*(1-(ms-self.bomb_lasttime)/self.bomb_invincible_time)/mpf
            pg.draw.circle(screen, cl.cyan, self.pos, bomb_radius, 2)
        elif self.hit_invincible:
            hit_radius = self.hit_invincible_time*(1-(ms-self.hit_lasttime)/self.hit_invincible_time)/mpf
            pg.draw.circle(screen, cl.cyan, self.pos, hit_radius, 2)

        angle_option = -(ms*36/100)%360
        d = 2*calc_d(angle_option)
        rot_option = pg.transform.rotate(self.option, angle_option)

        option_pos_lower = np.array([[+16, 32], [-16, 32], [+38, 16], [-38, 16]])*2
        option_pos_upper = np.array([[+8, -30], [-8, -30], [+24, -20], [-24, -20]])*2
        option_move = 4*mpf #4フレ分の時間
        dms = (ms-self.status.slow.last[self.status.slow.now])/option_move
        if 0 <= dms < 1:
            if self.status.slow.now:
                option_pos = dms*option_pos_upper+(1-dms)*option_pos_lower
            else:
                option_pos = dms*option_pos_lower+(1-dms)*option_pos_upper
        else:
            if self.status.slow.now:
                option_pos = option_pos_upper
            else:
                option_pos = option_pos_lower

        for o in option_pos:
            screen.blit(sp.eff_charge, self.pos-(32,32)+o, (0, 0, *sp.eff_charge.get_size()))
            screen.blit(rot_option, self.pos-(8*d, 8*d)+o, (0, 0, *rot_option.get_size()))


################ ?DA 弾幕 ################
class  AbstractBullet:
    def __init__(self, pos, radius) -> None:
        self.pos = pos; self.radius = radius
    def draw(self, screen: pg.Surface) -> None: pass
class CircleBullet(AbstractBullet):
    def __init__(self, pos, radius, border, color) -> None:
        super().__init__(pos=pos, radius=radius)
        self.border = border; self.color = color
    def draw(self, screen):
        pg.draw.circle(screen, cl.white, self.pos, self.radius+self.border)
        pg.draw.circle(screen, self.color, self.pos, self.radius+self.border, self.border)
class SmallCircleBullet(CircleBullet):
    def __init__(self, pos, color=cl.red):
        super().__init__(pos=pos, radius=2, border=2, color=color)
class MiddleCircleBullet(CircleBullet):
    def __init__(self, pos, color=cl.red):
        super().__init__(pos=pos, radius=10, border=4, color=color)
class StraightBullet(MiddleCircleBullet):
    def __init__(self, pos, direction, speed=15, color=cl.red):
        super().__init__(pos=pos, color=color)
        self.direction = direction
        self.speed = speed
    def update(self):
        self.pos = self.pos + self.speed*self.direction
class SquareBullet(AbstractBullet):
    def __init__(self, pos, radius):
        super().__init__(pos=pos, radius=radius)
    def draw(self, screen):
        pg.draw.rect(screen, cl.green, (self.pos[0]-self.radius, self.pos[1]-self.radius, self.radius*2, self.radius*2))
#        pg.draw.circle(screen, cl.cyan, self.pos, self.radius)
class ExpansionBullet(CircleBullet):
    def __init__(self, pos, color):
        self._radius = 10
        self._border = 4
        super().__init__(pos=pos, radius=self._radius, border=self._border, color=color)
        self.centercolor = cl.white
    def draw(self, screen):
        pg.draw.circle(screen, self.centercolor, self.pos, self.radius+self.border)
        pg.draw.circle(screen, self.color, self.pos, self.radius+self.border, self.border)
    def update_radius(self, radius):
        self.radius = radius
        self.border = round(self._border/self._radius*radius)
class Target(AbstractBullet):
    def __init__(self, pos, sprite, radius=0):
        super().__init__(pos, radius)
        self.sprite = sprite
    def draw(self, screen):
        size = self.sprite.get_size()
        screen.blit(self.sprite, self.pos-(size[0]/2, size[1]/2), (0, 0, *size))

################ ?SP スペルカード ################
class AbstractSpellCard:
    ""# docstringをスペカ名とする
    def __init__(self, t, ms, beats, game_step):
        self.t = t
        self.ms = ms
        self.beats = beats
        self.game_step = game_step
        self.phase = 0
        self.game_step.reimu.guard_spellcard(t, ms)
    def release(self, t, ms, beats): return [], []
    def finalize(self): pass
    def screen_out(self, bullets):
        tmp_bullets = list()
        for bullet in bullets:
            if PLAYAREA_RECT.collidepoint(*bullet.pos):
                tmp_bullets.append(bullet)
        return tmp_bullets

class OneSpellCard(AbstractSpellCard):
    """離陸「旅の始まり」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.bullets = list()
        self.margin = 200
        self.color_l = cl.red; self.color_r = cl.blue
        self.lc = np.array([PLAYAREA_RECT.left+self.margin, PLAYAREA_RECT.top+self.margin])
        self.rc = np.array([PLAYAREA_RECT.right-self.margin, PLAYAREA_RECT.top+self.margin])
        self.way = 20
    
    def bullets360(self, center, color=cl.red) -> List[AbstractBullet]:
        r = 2*np.pi*random()
        return [StraightBullet(center, circ((0,0),i*2*np.pi/self.way+r,1), color=color) for i in range(32)]

    def release(self, t, ms, beats):
        self.lcc = MiddleCircleBullet(self.lc, self.color_l)
        self.rcc = MiddleCircleBullet(self.rc, self.color_r)
        if beats[0] != beats[1]:
            for i in range(3):
                if beats.ignite(0,i,None,0):
                    self.bullets += self.bullets360(self.lc, self.color_l)
                if beats.ignite(2,i,None,0):
                    self.bullets += self.bullets360(self.rc, self.color_r)
            if beats.ignite(0,3,None,0):
                self.bullets += self.bullets360(self.lc, self.color_l)
            if beats.ignite(1,3,None,0):
                self.bullets += self.bullets360(self.lc, self.color_l)
            if beats.ignite(2,3,None,0):
                self.bullets += self.bullets360(self.rc, self.color_r)

        for bullet in self.bullets:
            bullet.update()
        return self.bullets+[self.lcc, self.rcc], []

class Hata1SpellCard(AbstractSpellCard): #左右に小回りして避ける
    """相似「葉脈標本」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.params = [pr.くるくる,pr.たちきの,pr.ひしがた,pr.はっぱや]
        self.offset = [-400, -150]
        self.T = round(len(self.params)/(BPM/(60*1000)))
        r = self.intp(0)
        self.bullets = [SmallCircleBullet(xy, rgb)for xy, rgb in zip(*self.hata_xyrgbs(r))]
        game_step.reimu.pos = PLAYAREA_CENTER.copy()
        self.game_step.reimu.controllable = False
        self.controllable_beat = count2beat(beat2count(beats[0])+2)
    def hata_xyrgbs(self, p: np.ndarray, calc_rgb=True) -> Tuple[List[Tuple[int,int]], Optional[List[Tuple[int,int,int]]]]:
        """パラメタpの畑写像から座標と色を返す"""
        zs = hatafast(p)
        xys = list()
        rgbs = list() if calc_rgb else None
        for i, z in enumerate(zs):
            if calc_rgb: rgb = hsv2rgb(i/len(zs), 1, 1)
            xy = z2xy(z)
            X = 2*np.array([self.offset[0]+220*i for i in range(3)])
            Y = 2*np.array(([self.offset[1]+110*i for i in range(4)]))
            for x in X:
                for y in Y:
                    _xy = (xy[0]+x, xy[1]+y)
                    xys.append(_xy)
                    if calc_rgb: rgbs.append(rgb)
        return xys, rgbs
    def intp(self, ms):
        ms -= self.t
        p = self.params[(ms//self.T)%len(self.params)]
        q = self.params[(ms//self.T+1)%len(self.params)]
        r = (1-ms%self.T/self.T)*p+ms%self.T/self.T*q
        return r
    def release(self, t, ms, beats):
        if beats.ignite(*self.controllable_beat):
            self.game_step.reimu.controllable = True
        if ms == self.ms: return self.bullets, []
        r = self.intp(ms)
        xys, _ = self.hata_xyrgbs(r, calc_rgb=False)
        tmp_bullets = list()
        for i, xy in enumerate(xys):
            self.bullets[i].pos = xy
            if PLAYAREA_RECT.collidepoint(*xy):
                tmp_bullets.append(self.bullets[i])
        return tmp_bullets, []

class Hata2SpellCard(Hata1SpellCard): #爆発するときに上が安置
    """相似「破られた手紙」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.params = [pr.かこまれ, pr.さんかく,pr.おてがみ,pr.せきへん]
class Hata3SpellCard(Hata1SpellCard): #難しい、小刻みに右斜め下にもぐりこむ
    """相似「龍の霊廟」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.params = [pr.みつびし,pr.ドラゴン,pr.くろすい,pr.ひびわれ]

class ExpansionSpellCard(AbstractSpellCard):
    """「膨張する時空間異常」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.exspeed = 0.2
        self.graze = 100
        self.pradius = PLAYAREA_RECT.height/4
        self.ppos = PLAYAREA_CENTER - (0, PLAYAREA_RECT.height/8)
        self.planet = CircleBullet(pos=self.ppos, radius=self.pradius, border=0, color=cl.white)
        self.game_step.reimu.pos = self.ppos.copy()
        self.gap = 100
        self.xmargin = 30
        self.ymargin = 0
        bulletss = list()
        y = PLAYAREA_RECT.top + self.ymargin
        while y < PLAYAREA_RECT.bottom - self.ymargin:
            x = PLAYAREA_RECT.left + self.xmargin
            tmp_bullets1 = list()
            while x < PLAYAREA_RECT.right - self.xmargin:
                tmp_bullets1.append(ExpansionBullet((x, y), cl.red))
                x += self.gap
            tmp_bullets2 = list()
            for bullet in tmp_bullets1[:-1]:
                _x = bullet.pos[0]+self.gap/2
                _y = bullet.pos[1]+self.gap*math.sqrt(3)/2
                tmp_bullets2.append(ExpansionBullet((_x, _y), cl.blue))
            bulletss.append(tmp_bullets1)
            bulletss.append(tmp_bullets2)
            y += self.gap*math.sqrt(3)
        self.bullets = sum(bulletss, [])

    def release(self, t, ms, beats):
        t -= self.t
        for bullet in self.bullets:
            if square_dist(self.game_step.reimu, bullet) < (self.game_step.reimu.radius+bullet.radius+self.graze)**2:
                bullet.update_radius(bullet.radius+self.exspeed)
                bullet.centercolor = (251, 180, 133)
            else:
                bullet.centercolor = cl.white
                
        return self.bullets, []

class OutOfSightSpellCard(AbstractSpellCard):
    """衝突「アウトオブサイト」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.game_step.reimu.pos = DEFAULT_POS.copy()
        self.exspeed = 0.2
        self.graze = 100
        self.pradius = PLAYAREA_RECT.height/4
        self.ppos = PLAYAREA_CENTER - (0, PLAYAREA_RECT.height/8)
        self.planet = CircleBullet(pos=self.ppos, radius=self.pradius, border=0, color=cl.white)
        self.gap = 100
        self.xmargin = 30
        self.ymargin = 0
        bulletss = list()
        y = PLAYAREA_RECT.top + self.ymargin
        while y < PLAYAREA_RECT.bottom - self.ymargin:
            x = PLAYAREA_RECT.left + self.xmargin
            tmp_bullets1 = list()
            while x < PLAYAREA_RECT.right - self.xmargin:
                tmp_bullets1.append(ExpansionBullet((x, y), cl.red))
                x += self.gap
            tmp_bullets2 = list()
            for bullet in tmp_bullets1[:-1]:
                _x = bullet.pos[0]+self.gap/2
                _y = bullet.pos[1]+self.gap*math.sqrt(3)/2
                tmp_bullets2.append(ExpansionBullet((_x, _y), cl.blue))
            bulletss.append(tmp_bullets1)
            bulletss.append(tmp_bullets2)
            y += self.gap*math.sqrt(3)
        self.bullets = [b for b in sum(bulletss, []) if self.pradius*0.7 < np.linalg.norm(self.ppos-b.pos)]
    def release(self, t, ms, beats):
        t -= self.t
        for bullet in self.bullets:
            if square_dist(self.game_step.reimu, bullet) < (self.game_step.reimu.radius+bullet.radius+self.graze)**2:
                bullet.update_radius(bullet.radius+self.exspeed)
                bullet.centercolor = (251, 180, 133)
            else:
                bullet.centercolor = cl.white
        if beat2ms((0,0,0,4)) <= ms:
            self.graze = float("inf")
            self.__doc__ = """「膨張する時空間異常」"""
        if beat2ms((0,3,3,3)) <= ms <= beat2ms((3,3,3,3)):
            return self.bullets+[self.planet], []
        else:
            return self.bullets, []

class GalaxySpellCard(AbstractSpellCard):
    """銀河「ライフゲーム」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.gap = PLAYAREA_RECT.width/13
        self.radius = self.gap/2
        self.grid = [[PLAYAREA_CENTER + (self.gap*x, self.gap*y) for x in range(-6, 7)]for y in range(-6, 7)]
        self.game_step.reimu.pos = PLAYAREA_CENTER.copy()
        self.game_step.reimu.controllable = False
        self.controllable_beat = count2beat(beat2count(beats[0])+4)
        self.gb = [list() for _ in range(8)]
        for i in range(8):
            for x in range(13):
                for y in range(13):
                    if GALAXY[i][x+1][y+1]:
                        self.gb[i].append(SquareBullet(self.grid[x][y], self.radius))
        self.bullets = self.gb[1]
        self.intps = list()
        self.planets = list()
        self.set_intp((0,0,3,0), 0,0,0)
        self.set_intp((0,2,3,0), 0,0,1)
        self.set_intp((0,0,0,1), 0,1,0)
        self.set_intp((0,2,0,1), 0,1,1)
        self.set_intp((0,0,1,1), 1,0,0)
        self.set_intp((0,2,1,1), 1,0,1)
        self.set_intp((0,0,2,1), 1,1,0)
        self.set_intp((0,2,2,1), 1,1,1)
        self.set_intp((0,0,3,1), 0,0,0); self.set_intp((0, 0, 3, 1), 1,0,0)
        self.set_intp((0,2,3,1), 0,0,1); self.set_intp((0, 2, 3, 1), 1,0,1)

        self.set_intp((0,0,0,2), 0,1,1); self.set_intp((0, 0, 0, 2), 1,1,0)
        self.set_intp((0,2,0,2), 0,1,0); self.set_intp((0, 2, 0, 2), 1,1,1)

        self.set_intp((0,0,1,2), 1,0,0); self.set_intp((0, 0, 1, 2), 0,0,0)
        self.set_intp((0,2,1,2), 1,0,1); self.set_intp((0, 2, 1, 2), 0,0,1)

    def set_intp(self, beat, vertical, from_up, up):
        start_beat = count2beat(beat2count(beat)-2)
        scale = [PLAYAREA_RECT.width, PLAYAREA_RECT.height][vertical]
        end_pos = PLAYAREA_CENTER-(2*up-1)*np.array([0, scale/4-self.gap/2][::2*vertical-1])
        start_pos = end_pos - (2*from_up-1)*np.array([scale/2, 0][::2*vertical-1])
        self.intps.append( Interpolation(start_beat, beat, start_pos, end_pos) )
        self.planets.append( CircleBullet(pos=start_pos, radius=scale/4-2, border=10, color=cl.red) )

    def release(self, t, ms, beats):
        if beats.ignite(*self.controllable_beat):
            self.game_step.reimu.controllable = True
        for i in range(4):
            for j in [0, 2]:
                if beats.ignite(i,j,None,None):
                    self.bullets = self.gb[(i+1)%8]
            for j in [1, 3]:
                if beats.ignite(i,j,None,None):
                    self.bullets = self.gb[(4+i+1)%8]
        for planet, intp in zip(self.planets, self.intps):
            planet.pos = intp(ms)
        scale = PLAYAREA_RECT.width/4
        radius = PLAYAREA_RECT.width/3-6
        if beats.ignite(0,3,1,2):
            self.planets.append( CircleBullet(pos=PLAYAREA_CENTER+(-scale, -scale), radius=radius, border=10, color=cl.red) )
        if beats.ignite(1,3,1,2):
            self.planets.append( CircleBullet(pos=PLAYAREA_CENTER+(scale, -scale), radius=radius, border=10, color=cl.red) )
        if beats.ignite(2,3,1,2):
            self.planets.append( CircleBullet(pos=PLAYAREA_CENTER+(scale, scale), radius=radius, border=10, color=cl.red) )
        if beats.ignite(3,3,1,2):
            self.planets.append( CircleBullet(pos=PLAYAREA_CENTER+(-scale, scale), radius=radius, border=10, color=cl.red) )

        return self.bullets+self.planets, []

class FlipSpellCard(AbstractSpellCard):
    """逆転「空間識失調」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.mybullet =  lambda pos, color: CircleBullet(pos, radius=10, border=10, color=color)
        self.earch_radius = 100
        self.way = 32
        self.bullets = list()
        self.blackhole = Target(PLAYAREA_CENTER, sp.houi2, self.earch_radius)
        self.game_step.reimu.controllable = False
        self.controllable_beat = count2beat(beat2count(beats[0])+2)
    def release(self, t, ms, beats):
        if beats.ignite(*self.controllable_beat):
            self.game_step.reimu.controllable = True
        if beats.ignite(0, 0, 1, None,):
            self.game_step.reimu.pos = DEFAULT_POS.copy()
            self.game_step.flip_x = False 
            self.game_step.flip_y = True
        if beats.ignite(0, 0, 2, None,):
            self.game_step.reimu.pos = DEFAULT_POS.copy()
            self.game_step.flip_x = True 
            self.game_step.flip_y = True
        if beats.ignite(0, 0, 3, None,):
            self.game_step.reimu.pos = DEFAULT_POS.copy()
            self.game_step.flip_x = False
            self.game_step.flip_y = False
        if beats.ignite(0, 0, 0, None,):
            self.game_step.reimu.pos = DEFAULT_POS.copy()
            self.game_step.flip_x = True 
            self.game_step.flip_y = False
        beat = beats[0]
        travel = (DEFAULT_POS-PLAYAREA_CENTER)[1]-self.earch_radius
        if beats.ignite(0,None,None,None) or beats.ignite(2,None,None,None):
            for i in range(self.way):
                start_pos = circ(PLAYAREA_CENTER, i*2*np.pi/self.way+np.pi/2+np.pi/self.way, self.earch_radius)
                end_pos = circ(PLAYAREA_CENTER, i*2*np.pi/self.way+np.pi/2+np.pi/self.way, travel)
                color = hsv2rgb(-(i+1)*(self.way//2+1)%self.way/self.way, 1, 1)
                bullet = self.mybullet(start_pos, color)
                bullet.intp = Interpolation(beat, count2beat(beat2count(beat)+4), start_pos, end_pos)
                self.bullets.append(bullet)
        elif beats.ignite(1,None,None,None) or beats.ignite(3,None,None,None):
            for i in range(self.way):
                start_pos = circ(PLAYAREA_CENTER, i*2*np.pi/self.way+np.pi/2, self.earch_radius)
                end_pos = circ(PLAYAREA_CENTER, i*2*np.pi/self.way+np.pi/2, travel)
                color = hsv2rgb(-(i+1)*(self.way//2+1)%self.way/self.way, 1, 1)
                bullet = self.mybullet(start_pos, color)
                bullet.intp = Interpolation(beat, count2beat(beat2count(beat)+4), start_pos, end_pos)
                self.bullets.append(bullet)
        tmp_bullets = list()
        for b in self.bullets:
            b.pos = b.intp(ms)
            if PLAYAREA_RECT.collidepoint(*b.pos):
                tmp_bullets.append(b)
        self.bullets = tmp_bullets
        return self.bullets+[self.blackhole], []
    def finalize(self):
        self.game_step.flip_x = False
        self.game_step.flip_y = False


class AmeSpellCard(AbstractSpellCard):
    """無月「天泣の涙雨」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.bullets = list()
        self.graze = 200 
        self.yoke = 0.002
        self.d_kasa = np.array([0, -300])
    def release(self, t, ms, beats):
        for _ in range(4):
            pos = np.array((PLAYAREA_RECT.width*random(), PLAYAREA_RECT.top))
            direction = circ((0,0), np.pi/2+np.pi/16*(random()-.5),1)
            speed = 10+10*random()
            bullet = StraightBullet(pos, direction=direction, speed=speed)
            bullet.radius = 4
            bullet.color = cl.blue
            self.bullets.append(bullet)
        for bullet in self.bullets:
            bullet.update()
            kasa = self.game_step.reimu.pos+self.d_kasa
            if np.linalg.norm(kasa-bullet.pos) < self.graze:
                dx = self.yoke*(bullet.pos-kasa)[0]
                bullet.direction += (dx, 0)
            if square_dist(self.game_step.reimu, bullet) < 100:
                bullet.pos = (0,0)
        self.bullets = self.screen_out(self.bullets)
        return self.bullets, []


class LastSpellCard(AbstractSpellCard):
    """着陸「431光年の旅路」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        self.center1 = PLAYAREA_CENTER
        self.center2 = np.array([PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.top-100])
        self.way = 18 
        self.earth_radius = 50
        self.cspeed = 0.0005
        self.const = 1
        self.gap = 70
        self.gain = 2*self.cspeed/1000
        self.gain2 = self.gap/1000
        self.game_step.reimu.pos = DEFAULT_POS.copy()
        self.game_step.reimu.bomb_stock = 0
        self.e = Target(self.center2, sp.etama7a)
        self.y = Target(self.center2, sp.yukari_bs)
    def release(self, t, ms, beats):
        t -= self.t
        T = 60*10
        if T < t <= 1.5*T:
            self.center = self.center1
            self.cspeed += self.gain
            self.gap -= self.gain2
        elif 1.5*T < t:
            self.center = self.center1
        else:
            self.center = t/T*self.center1 + (1-t/T)*self.center2
        self.lwind = list()
        self.rwind = list()
        for j in range(1, 12):
            for i in range(32):
                self.lwind.append(
                    MiddleCircleBullet(circ(self.center, (j+self.const)*self.cspeed*t+i*2*np.pi/self.way, self.earth_radius+self.gap*j), cl.red)
                )
                self.rwind.append(
                    MiddleCircleBullet(circ(self.center, -(j+self.const)*self.cspeed*t-i*2*np.pi/self.way, self.earth_radius+self.gap*j+self.gap*.5), cl.blue)
                )
        self.e.pos = self.center
        MS = beat2ms((0, 2, 0, 0))
        self.y.pos = self.center + (0, 6*np.sin(2*np.pi*(ms/MS))-40)
        return self.lwind+self.rwind, [self.e, self.y]
class HiddenCommandSpellCard(AbstractSpellCard):
    """「幸せになれる隠しコマンド」"""
    def __init__(self, t, ms, beats, game_step):
        super().__init__(t, ms, beats, game_step)
        Q = 0.7071067811865475244008
        TELEPORT = 30
                    #1234123412341234
        self.notes = "msummsmmuusshmhm"*3
        self.m = MiddleCircleBullet( PLAYAREA_CENTER+(+1*TELEPORT, +0*TELEPORT) )
        self.h = MiddleCircleBullet( PLAYAREA_CENTER+(-1*TELEPORT, +0*TELEPORT) )
        self.u = MiddleCircleBullet( PLAYAREA_CENTER+(+0*TELEPORT, -1*TELEPORT) )
        self.s = MiddleCircleBullet( PLAYAREA_CENTER+(+0*TELEPORT, +1*TELEPORT) )
        self.L = MiddleCircleBullet( PLAYAREA_CENTER+(-Q*TELEPORT, -Q*TELEPORT) )
        self.R = MiddleCircleBullet( PLAYAREA_CENTER+(+Q*TELEPORT, -Q*TELEPORT) )
        self.l = MiddleCircleBullet( PLAYAREA_CENTER+(-Q*TELEPORT, +Q*TELEPORT) )
        self.r = MiddleCircleBullet( PLAYAREA_CENTER+(+Q*TELEPORT, +Q*TELEPORT) )
        self.game_step.reimu.pos = PLAYAREA_CENTER.copy()
    def release(self, t, ms, beats):
        if beats.ignite(None, None, None, None):
            self.game_step.reimu.pos = PLAYAREA_CENTER.copy()
        for i, x in enumerate(self.notes):
            if beat2count(beats[0]) == beat2count((2,0,1,0))+i:
                tmp = {getattr(self, y) for y in "mhusLRlr"}
                tmp.discard(getattr(self, x))
                return tmp, []
        if beats.ignite(None, None, None, None):
            self.game_step.reimu.pos = PLAYAREA_CENTER.copy()
        return {getattr(self, x) for x in "mhusLRlr"}, []

################ ?TI タイムスケジュール ################
TIME_SCHEDULE = Dict({k:v for k, v in {
    (0,0,0,0): HiddenCommandSpellCard,#開始
}.items() if v is not None})


################ ?ST プレイ中の各ステップ ################
class AbstractStep:
    def __init__(self, screen_display: pg.Surface, clock: pg.time.Clock) -> None:
        self.screen_display = screen_display; self.clock = clock
    def play(self, t: int, controller_input: Dict) -> None: pass

class GameStep(AbstractStep):
    def __init__(self, screen_display: pg.Surface, clock: pg.time.Clock) -> None:
        super().__init__(screen_display, clock)
        self.screen_playarea = pg.Surface((PLAYAREA_RECT.width, PLAYAREA_RECT.height))
        self.screen_playarea_color = cl.black
        self.info = Dialog(INFO_RECT, border=2, margin=5)
        dialog_size = np.array([530, 200])
        dialog_pos = DISPLAY_CENTER-dialog_size/2 + (-60, 300)
        self.dialog = Dialog(pg.Rect(*dialog_pos, *dialog_size))
        self.reimu = Reimu()

        self.screen_display.blit(sp.bg, (0,0))
        self.screen_display.blit(sp.logo, (DISPLAY_RECT.right-444, DISPLAY_RECT.bottom-390))
        self.gameover = False
        self.beats = Beats([None, None])
        self.spell_card = AbstractSpellCard(0, 0, self.beats, self)
        self.flip_x = False
        self.flip_y = False
        self.laspetxt = Target(PLAYAREA_CENTER, sp.laspetxt)
        self.clear = False
        self.i = 0
        self.txts = [
            ("…はじめまして？", cl.purple),
            ("うわっ、ほんとに居た！妖怪かしら？\n…でもその声はずっと聴こえていた歌と一緒ね。", cl.red),
            ("わたしはアンドロイドです、\nアンドロイドの「結月ゆかり」です。\n\n100年ぶりに人間に逢えたわ。\nずっと孤独だったの。", cl.purple),
            ("なんだってこんな場所に居るのよ？\n", cl.red),
            ("それはこの星に不時着してしまって…\n救難信号として歌をずっと歌っていたの。\n\n途中諦めかけたけど、\n無駄じゃなかったってことね。", cl.purple),
            ("それは…なかなか凄まじいわね、\n\n私も結構酷い目にあってるけど方だけど、\nそんな経験はなかなか無いわ。", cl.red),
            ("貴方は動く星々を避けながら、\n最後まで私の歌を聞いてくれたのね。\n\nすごく嬉しいわ…\n星々を避けることがすごく上手なのね。", cl.purple),
            ("そうね？普段から\n似たような事をやっているからかしら？\nたぶん避ける事なら宇宙で一番は私よ！", cl.red),
            ("申し遅れたけど、私は「博麗霊夢」、地球にある\n幻想郷って場所で巫女をやっているわ。\n\nしかし、ここまで来るのは大変だったわ。\n何よあの弾幕は！", cl.red),
            ("ここの周りの星々は、\n不安定な重力場が互いに影響しあっていて\n複雑でカオスな軌道を動くの。", cl.purple),
            ("私達がいるこの星だけなのよ、\n不動点＜fixed point＞なのは。そのせいで\n私はここからずっと脱出できないでいたの。", cl.purple),
            ("なるほど…\nだから宇宙から歌が聴こえていたのね、\n謎が解けてスッキリしたわ。\n\n…それで、アンタはこれからどうするのよ？", cl.red),
            ("そうね…今まで諦めていたけど、\n貴方についていったら、\n私も一緒に脱出できるかもしれない！\n\n勿論、貴方が良ければだけど…", cl.purple),
            ("こうなる展開は正直薄々予想してたわ。\n\nしょうがないわねぇ…\nしっかりつかまってなさいよ？\n振り落とされないようにね！", cl.red),
            ("地球に帰還します、このまま進んでください。", cl.white),
        ]

    def sc(self, spell_card):
        self.spell_card.finalize()
        self.spell_card = spell_card

    def play(self, t: int, controller_input: Dict) -> None:
        self.screen_playarea.fill(self.screen_playarea_color)
        self.info.start(cl.black)
        ms = pg.mixer.music.get_pos() + CONFIG["ms"]
        self.beats = Beats([ms2beat(ms), self.beats[0]])

        for k, v in TIME_SCHEDULE.items():
            if v is None: continue
            if self.beats.ignite(*k):
                self.sc(v(t, ms, self.beats, self))
        if self.beats.ignite(0,2,3,8): #終了
            self.reimu.controllable = False
            self.clear = True

        bullets, targets = self.spell_card.release(t, ms, self.beats)
        for target in targets:
            target.draw(self.screen_playarea)
        is_hit = False
        for bullet in bullets:
            if check_hit(self.reimu, bullet) and not CONFIG["invincible"]:
                is_hit = True; break
        self.reimu.update(t, ms, controller_input, is_hit)
        self.reimu.draw_lower(t, ms, self.screen_playarea)
        for bullet in bullets:
            bullet.draw(self.screen_playarea)
        self.reimu.draw_upper(t, ms, self.screen_playarea)

        if beat2ms((0,0,3,7)) < ms <= beat2ms((0,2,3,7)):
            self.laspetxt.draw(self.screen_playarea)

        # テキスト描画処理
        self.fontoffset = 0
        self.info.print(f"{self.spell_card.__doc__}")
        self.info.print(f"{self.reimu.bomb_stock}{'★'*self.reimu.bomb_stock}", color=cl.green)
        self.info.print(f"fps:{self.clock.get_fps():.2f}")
        self.info.print(f"弾数: {len(bullets)}")
        self.info.print(f"■□♡♥☆★こんにちわ世界")
        self.info.print(f"{ms}")
        self.info.print(f"{self.beats[0]}")
        for s in beat2squares(self.beats[0]):
            self.info.print(s)

        if self.flip_x or self.flip_y:
            screen_playarea_flipped = pg.transform.flip(self.screen_playarea, self.flip_x, self.flip_y)
            self.screen_display.blit(screen_playarea_flipped, (BORDER_RECT.left, BORDER_RECT.top))
        else:
            self.screen_display.blit(self.screen_playarea, (BORDER_RECT.left, BORDER_RECT.top))
        self.info.end(self.screen_display)
#        self.screen_display.blit(self.dialog, (INFO_RECT.left, INFO_RECT.top))
        pg.draw.rect(self.screen_display, cl.green, BORDER_RECT, 2)
        if self.clear:
            display_l(self.screen_display, sp.reimu_l)
            display_r(self.screen_display, sp.yukari_r)
            self.dialog.start(cl.black)
            self.dialog.print(*self.txts[self.i])
            self.dialog.end(self.screen_display)
    
class TitleStep(AbstractStep):
    def __init__(self, screen_display: pg.Surface, clock: pg.time.Clock) -> None:
        super().__init__(screen_display, clock)
        self.offset = 0
        self.scroll_speed = 20
        self.scroll_length= 620
        self.margin_y = 150

    def play(self, t: int, controller_input: Dict) -> None:
        if controller_input.down:
            self.offset -= self.scroll_speed
        elif controller_input.up:
            self.offset += self.scroll_speed
        if self.offset < -(self.scroll_length+self.margin_y): self.offset = -(self.scroll_length+self.margin_y)
        if self.offset > 0: self.offset = 0
        self.screen_display.blit(sp.title_bg, (0,0))
        display_l(self.screen_display, sp.reimu_l)
        display_r(self.screen_display, sp.marisa_r)
        self.screen_display.blit(sp.description, (350, self.margin_y + self.offset))

class ConfigStep(AbstractStep):
    def __init__(self, screen_display: pg.Surface, clock: pg.time.Clock) -> None:
        super().__init__(screen_display, clock)
        dialog_size = np.array([500, 500])
        dialog_pos = DISPLAY_CENTER-dialog_size/2 + (0, 200)
        self.dialog = Dialog(pg.Rect(*dialog_pos, *dialog_size))
        self.txts = [
            "ボムを捨てて挑むって…\n不可能<インポッシブル>だぜ…\n馬鹿だろお前(呆れ)",
            "自前のボムだけで挑むなんて…\n狂気<ルナティック>だぜ…\n人間やめるのか？",
            "そんなボム数で大丈夫か？\n難関<ハード>だぜ…\nできたら尊敬するぜ!",
            "私のボム持ってけよ、\n普段<ノーマル>の実力\n見せて欲しいのぜ!",
            "そんなに持ってったら\n私が破産<イージー>\nしちゃうのぜ(泣)"
        ]
        self.skip = 0
        self.fontoffset = 20
    def play(self, t: int, controller_input: Dict) -> None:
        self.screen_display.blit(sp.witch1, (0,0))
        self.dialog.start(cl.black)
        self.dialog.print("で魔理沙からボムを貰おう!")
        self.dialog.print(f"{CONFIG.bomb_stock}{'★'*CONFIG.bomb_stock}", color=cl.green)
        self.dialog.print()
        if CONFIG.bomb_stock == 0:
            self.dialog.print(self.txts[0], color=cl.yellow)
        elif CONFIG.bomb_stock in range(1, LEVEL[0]):
            self.dialog.print(self.txts[1], color=cl.yellow)
        elif CONFIG.bomb_stock in range(4, LEVEL[1]):
            self.dialog.print(self.txts[2], color=cl.yellow)
        elif CONFIG.bomb_stock in range(7, LEVEL[2]):
            self.dialog.print(self.txts[3], color=cl.yellow)
        else:
            self.dialog.print(self.txts[4], color=cl.yellow)

        self.dialog.print()
        self.dialog.print("で八雲紫の力でワープしよう!")
        self.dialog.print(f"{self.skip} {TIME_SCHEDULE.at(self.skip).__doc__}", color=cl.green)
        self.dialog.print()
        if self.skip == 0:
            self.dialog.print(f"こんなん自力で十分よ！", color=cl.red)
            self.dialog.print()
        else:
            self.dialog.print(f"癪だけど紫に頼むか…", color=cl.red)
            self.dialog.print(f"宇宙にスキマって無茶かしら?", color=cl.red)
        self.dialog.print()
        if check_good_ending():
            self.dialog.print("グッドエンディングに到達可能")
        else:
            self.dialog.print("練習用モードです")
        display_l(self.screen_display, sp.reimu_l)
        display_r(self.screen_display, sp.marisa_r)
        self.dialog.end(self.screen_display)


################ ?MA メインループ ################
class MainLoop:
    def __init__(self) -> None:
        pg.init()
        pg.display.set_caption("東方不動点")
        self.screen_display = pg.display.set_mode((DISPLAY_RECT.width, DISPLAY_RECT.height))
        self.controller = MyController()
        self.clock = pg.time.Clock()
        self.current_step = TitleStep
        se.load()
        sp.load()
        pg.mixer.music.load("data/Yours.wav")
        pg.mixer.music.play(loops=-1)

    def restart_game(self) -> None:
        self.current_step = GameStep
        pg.mixer.music.load("data/幸せになれる隠しコマンドがあるらしい.ogg")
        pg.mixer.music.stop()
        pg.mixer.music.rewind()
        pg.mixer.music.play(-1, CONFIG["ms"]/1000)
        self.game_step = GameStep(self.screen_display, self.clock)
        self.t = 0
        se.ok00.play()
    
    def back_to_title(self) -> None:
        self.current_step = TitleStep
        pg.mixer.music.load("data/Yours.ogg")
        pg.mixer.music.stop()
        pg.mixer.music.rewind()
        pg.mixer.music.play(loops=-1)
        self.title_step = TitleStep(self.screen_display, self.clock)
        se.cancel00.play()

    def go_to_config(self) -> None:
        self.current_step = ConfigStep
        pg.mixer.music.load("data/IMP.ogg")
        pg.mixer.music.stop()
        pg.mixer.music.rewind()
        pg.mixer.music.play(loops=-1)
        self.config_step = ConfigStep(self.screen_display, self.clock)
        se.extend.play()

    def main(self) -> None:
        self.game_step = GameStep(self.screen_display, self.clock)
        self.title_step = TitleStep(self.screen_display, self.clock)
        self.config_step = ConfigStep(self.screen_display, self.clock)
        self.t = 0
        while True:
            controller_input = self.controller.check_input()
            if self.current_step == GameStep:
                self.game_step.play(self.t, controller_input)
                if (self.game_step.reimu.bomb_stock < 0
                    and not self.game_step.reimu.bomb_invincible
                    and not self.game_step.reimu.hit_invincible
                ):
                    self.back_to_title()
            elif self.current_step == TitleStep:
                self.title_step.play(self.t, controller_input)
            elif self.current_step == ConfigStep:
                self.config_step.play(self.t, controller_input)
            else:
                print("game_step error")
            self.clock.tick(60)
            pg.display.update()

            for event in pg.event.get():
                if event.type == QUIT:
                    pg.quit(); sys.exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_r:
                        self.restart_game()
                    if event.key == pg.K_ESCAPE:
                        self.back_to_title()
                    if event.key == pg.K_c and self.current_step == TitleStep:
                        self.go_to_config()
                    if self.current_step == GameStep:
                        if self.game_step.clear:
                            if event.key == pg.K_LEFT:
                                self.game_step.i -= 1
                                if self.game_step.i < 0: self.game_step.i = 0
                            if event.key == pg.K_RIGHT:
                                self.game_step.i += 1
                                if len(self.game_step.txts)-1 < self.game_step.i:
#                                    self.game_step.i = len(self.game_step.txts)-1
                                    self.back_to_title()
                    if self.current_step == ConfigStep:
                        if event.key == pg.K_LEFT:
                            CONFIG["bomb_stock"] -= 1
                            if CONFIG["bomb_stock"] < 0: CONFIG["bomb_stock"] = 0
                            se.ok00.play()
                        if event.key == pg.K_RIGHT:
                            CONFIG["bomb_stock"] += 1
                            se.ok00.play()
                        if event.key == pg.K_UP:
                            self.config_step.skip = (self.config_step.skip-1)%len(TIME_SCHEDULE)
                        if event.key == pg.K_DOWN:
                            self.config_step.skip = (self.config_step.skip+1)%len(TIME_SCHEDULE)
                        if event.key in [pg.K_UP, pg.K_DOWN]:
                            for i, (k, v) in enumerate(TIME_SCHEDULE.items()):
                                if i == self.config_step.skip:
                                    CONFIG["beat"] = k
                                    CONFIG["ms"] = math.ceil(beat2ms(CONFIG["beat"]))
                            se.ok00.play()
            self.t += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beat', type=str, default="0000")
    parser.add_argument('--bomb_stock', type=int, default=9)
    parser.add_argument('--invincible', type=bool, default=False)
    CONFIG = Dict(parser.parse_args().__dict__)
    CONFIG["bomb_stock"] = CONFIG.bomb_stock
    CONFIG["beat"] = tuple(map(int, CONFIG.beat))
    CONFIG["ms"] = math.ceil(beat2ms(CONFIG["beat"]))
    print(CONFIG["beat"], CONFIG["ms"], ms2beat(CONFIG["ms"]), ms2beat(math.floor(CONFIG["ms"])), ms2beat(math.ceil(CONFIG["ms"])), ms2beat(round(CONFIG["ms"])))
    assert CONFIG["beat"] == ms2beat(CONFIG["ms"])
    main_loop = MainLoop()
    main_loop.main()