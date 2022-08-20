import pygame as pg; from pygame.locals import *
import numpy as np
from addict import Dict as addict_dict
import colorsys
from typing import *
import sys
import math
import argparse
from random import random


class se: pass #se用の名前空間
class Dict(addict_dict):
    def at(self, k: int) -> Any:
        return self[list(self.keys())[k]]

CONFIG = Dict()
BPM = 190
DISPLAY_RECT = pg.Rect(0, 0, 1280, 960)
BORDER_RECT = pg.Rect(62, 30, 834-62, 930-30)
PLAYAREA_RECT = pg.Rect(0, 0, 2*386, 2*450)
INFO_RECT = pg.Rect(834+30, 30, 1280-834-2*30, 500)

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

################ 補助関数群 ################
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
    return math.cos(angle%90/360*2*math.pi)+math.sin(angle%90/360*2*math.pi)

# beat(拍数)関連の補助関数
beat_t = Tuple[int, int, int, int]
def ms2beat(ms: int) -> beat_t:
    count = int(BPM/(60*1000)*ms)
    beat = (count%4, count//4%4, count//16%4, count//64)
    assert beat[0]+4*beat[1]+16*beat[2]+64*beat[3] == count
    return beat
def beat2ms(beat: beat_t) -> int:
    count = beat[0]+4*beat[1]+16*beat[2]+64*beat[3]
    ms = count/(BPM/(60*1000))
    return ms
def beat2squares(beat: beat_t) -> List[str]:
    squares = [["□"]*4 for _ in range(5)]
    for i in range(4):
        squares[i][beat[i]%4] = "■"
    squares[4][beat[i]//4%4] = "■"
    return reversed(["".join(s) for s in squares])

def check_hit(a, b) -> bool:
    """被弾判定、東方は円どうしの交点が直交するまで重なるが特に考慮せずに通常の判定"""
    return (a.pos[0]-b.pos[0])**2 + (a.pos[1]-b.pos[1])**2 < (a.radius+b.radius)**2

################ メインクラス群 ################
class Status:
    """時機の現在の状態"""
    KEY = ["left", "right", "up", "down", "slow", "shot", "bomb"]
    def __init__(self) -> None:
        self.now = False
        self.last = [-float("inf"), -float("inf")] # 負の無限時間から定常状態

class Beats(list):
    """現在の拍子を管理するクラス、長さ2のリストと同様"""
    def ignite(self, *beat: beat_t) -> bool:
        """現フレームで指定した拍子に到達したときに発火する"""
        return self[0] != self[1] and self[0] == beat

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
        )

class Reimu:
    """自機のクラス"""
    def __init__(self) -> None:
        self.pl00 = pg.image.load("data/w2x_pl00.png")
        self.pl10 = pg.image.load("data/w2x_pl10.png")
        self.pos = np.array((PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.bottom+30))
        self.option = pg.Surface((32, 32), pg.SRCALPHA)
        self.option.blit(self.pl00, (0, 0), (3*32*2, 3*48*2, 16*2, 16*2))
        self.sloweffect = pg.Surface((128, 128), pg.SRCALPHA)
        self.sloweffect.blit(pg.image.load("data/w2x_eff_sloweffect.png"), (0, 0), (0,0, 128, 128))
        self.eff_charge = pg.Surface((64, 64), pg.SRCALPHA)
        self.eff_charge.blit(pg.image.load("data/w2x_eff_charge.png"), (0, 0), (0,0, 128, 128))
        self.bomb_stock = CONFIG["bomb_stock"]
        self.bomb_invincible = False
        self.bomb_invincible_time = 2*60 # ボム無敵時間
        self.bomb_lasttime = float("inf")
        self.hit_invincible = False
        self.hit_invincible_time = 1*60 # 被弾無敵時間
        self.spellcard_invincible = False
        self.spellcard_invincible_time = 0.5*60 # スペルカード移行時無敵
        self.hit_lasttime = float("inf")
        self.radius = 5 # 通常は3.0, 2.4の2倍
        self.kurai_bomb = 30 # 通常は8、15でもよさそう
        self.status = Dict({k:Status() for k in Status.KEY})

    def update(self, t: int, controller_input: Dict, is_hit: bool) -> None:
        """自機の位置、無敵等をupdate"""
        # controller inputとreimu.statusが変化したら更新し、更新時刻を記録する
        for k in Status.KEY:
            if self.status[k].now^controller_input[k]:
                self.status[k].last[controller_input[k]] = t
                self.status[k].now = controller_input[k]

        speed = 2*[4.0, 2.0][controller_input.slow]
        if [controller_input.left, controller_input.right, controller_input.up, controller_input.down].count(True) == 2:
            speed /= math.sqrt(2)

        if controller_input.left:    self.pos[0] -= speed
        elif controller_input.right: self.pos[0] += speed
        if controller_input.up:      self.pos[1] -= speed
        elif controller_input.down:  self.pos[1] += speed
        
        # 画面外脱出防止
        mergin = 40
        if self.pos[0] < PLAYAREA_RECT.left+mergin:   self.pos[0] = PLAYAREA_RECT.left+mergin
        if self.pos[0] > PLAYAREA_RECT.right-mergin:  self.pos[0] = PLAYAREA_RECT.right-mergin
        if self.pos[1] < PLAYAREA_RECT.top+mergin:    self.pos[1] = PLAYAREA_RECT.top+mergin
        if self.pos[1] > PLAYAREA_RECT.bottom-mergin: self.pos[1] = PLAYAREA_RECT.bottom-mergin

        # 通常のボムの処理
        if ( self.status["bomb"].now and 0 < self.bomb_stock 
            and not self.bomb_invincible and not self.hit_invincible ):
            self.bomb_invincible = True
            self.bomb_stock -= 1
            self.bomb_lasttime = t
            se.slash.play()

        # 通常の被弾の処理
        elif is_hit and not self.bomb_invincible and not self.hit_invincible and not self.spellcard_invincible:
            self.hit_invincible = True
            self.bomb_stock -= 2
            self.hit_lasttime = t
            se.pldead00.play()

        # 喰らいボムの処理
        elif ( self.status["bomb"].now and 0 < self.bomb_stock and self.hit_invincible ): 
            if t-self.hit_lasttime <= self.kurai_bomb:
                self.hit_invincible = False
                self.bomb_invincible = True
                self.bomb_stock += 1
                self.bomb_lasttime = t
                se.slash.play()

        # ボム不可時の処理
        elif ( self.status["bomb"].now and 0 >= self.bomb_stock 
            and not self.bomb_invincible and not self.hit_invincible ):
            se.invalid.play()

        # 無敵時間終了処理
        if 0 < t-self.bomb_lasttime-self.bomb_invincible_time:
            self.bomb_invincible = False
        if 0 < t-self.hit_lasttime-self.hit_invincible_time:
            self.hit_invincible = False
        if 0 < t-self.hit_lasttime-self.hit_invincible_time:
            self.hit_invincible = False
        if 0 < t-self.spellcard_lasttime-self.spellcard_invincible_time:
            self.spellcard_invincible = False
    
    def guard_spellcard(self, t: int) -> None:
        self.spellcard_invincible = True
        self.spellcard_lasttime = t

    def draw(self, t: int, screen: pg.Surface) -> None:
        """Surfaceへの自機の描画"""
        if self.status.left.now: reimu_offset = 1
        elif self.status.right.now: reimu_offset = 2
        else: reimu_offset = 0

        screen.blit(
            self.pl10 if any([self.bomb_invincible, self.hit_invincible, self.spellcard_invincible]) else self.pl00,
            self.pos-(16*2, 24*2), (0+32*(t//8%4)*2, reimu_offset*48*2, 32*2, 48*2))
        if self.bomb_invincible:
            bomb_radius = self.bomb_invincible_time*(1-(t-self.bomb_lasttime)/self.bomb_invincible_time)
            pg.draw.circle(screen, cl.cyan, self.pos, bomb_radius, 2)
        elif self.hit_invincible:
            hit_radius = self.hit_invincible_time*(1-(t-self.hit_lasttime)/self.hit_invincible_time)
            pg.draw.circle(screen, cl.cyan, self.pos, hit_radius, 2)

        angle_option = -t*6%360
        d = 2*calc_d(angle_option)
        rot_option = pg.transform.rotate(self.option, angle_option)

        angle_sloweffect = t*3%360
        d2 = 2*calc_d(angle_sloweffect)
        rot_sloweffect = pg.transform.rotate(self.sloweffect, angle_sloweffect)
        rot_sloweffect2 = pg.transform.rotate(self.sloweffect, -angle_sloweffect)

        option_pos_lower = np.array([[+16, 32], [-16, 32], [+38, 16], [-38, 16]])*2
        option_pos_upper = np.array([[+8, -30], [-8, -30], [+24, -20], [-24, -20]])*2
        option_move = 4
        dt = (t-self.status.slow.last[self.status.slow.now])/option_move
        if 0 <= dt < 1:
            if self.status.slow.now:
                option_pos = dt*option_pos_upper+(1-dt)*option_pos_lower
            else:
                option_pos = dt*option_pos_lower+(1-dt)*option_pos_upper
        else:
            if self.status.slow.now:
                option_pos = option_pos_upper
            else:
                option_pos = option_pos_lower

        for o in option_pos:
            screen.blit(self.eff_charge, self.pos-(32,32)+o, (0, 0, *self.eff_charge.get_size()))
            screen.blit(rot_option, self.pos-(8*d, 8*d)+o, (0, 0, *rot_option.get_size()))

        if self.status.slow.now:
            screen.blit(rot_sloweffect, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))
            screen.blit(rot_sloweffect2, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))

class GameStep:
    def __init__(self, screen_display: pg.Surface, clock: pg.time.Clock) -> None:
        self.screen_display = screen_display
        self.clock = clock
        self.screen_playarea = pg.Surface((PLAYAREA_RECT.width, PLAYAREA_RECT.height))
        self.screen_playarea_color = cl.gray1
        self.screen_info = pg.Surface((INFO_RECT.width, INFO_RECT.height))
        self.screen_info_color = cl.darkblue
        self.fontsize = 30
        self.fontoffset = 0
        self.fontname = "malgungothic"
        self.font = pg.font.SysFont(self.fontname, self.fontsize)
        self.bg = pg.image.load("data/bg.png")
        self.logo = pg.image.load("data/logo.png")
        self.reimu = Reimu()

        self.screen_display.blit(self.bg, (0,0))
        self.screen_display.blit(self.logo, (DISPLAY_RECT.right-444, DISPLAY_RECT.bottom-390))
        self.gameover = False
        self.beats = Beats([None, None])
        self.spell_card = AbstractSpellCard(0, 0, self.beats, self.reimu)

    def play(self, t: int, controller_input: Dict) -> None:
        self.screen_playarea.fill(self.screen_playarea_color)
        self.screen_info.fill(self.screen_info_color)
        # テキスト描画処理
        self.fontoffset = 0
        self.print(f"{self.reimu.bomb_stock}{'★'*self.reimu.bomb_stock}")
        self.print(f"fps:{self.clock.get_fps():.2f}")
        self.print(f"hit: {self.reimu.hit_invincible, t-self.reimu.hit_lasttime-self.reimu.hit_invincible_time}")
        self.print(f"bomb: {self.reimu.bomb_invincible, t-self.reimu.bomb_lasttime-self.reimu.bomb_invincible_time}")
        self.print(f"■□♡♥☆★")
        self.print("こんにちわ世界")
        ms = pg.mixer.music.get_pos() + CONFIG["ms"]
        self.beats = Beats([ms2beat(ms), self.beats[0]])
        self.print(f"{ms}")
        self.print(f"{self.beats[0]}")
        self.print(f"{self.fontname}")
        for s in beat2squares(self.beats[0]):
            self.print(s)

        # ?SPELLCARD TIME SCHEDULE
        if self.beats.ignite(0,0,0,0) and ms != -1: #再生終了で戻るのを防ぐ
            self.reimu.guard_spellcard(t)
            self.spell_card = OneSpellCard(t, ms, self.beats, self.reimu)
        elif self.beats.ignite(0,0,1,0):
            self.reimu.guard_spellcard(t)
            self.spell_card = HataSpellCard(t, ms, self.beats, self.reimu)
        elif self.beats.ignite(0,0,2,2): #1サビ
            self.reimu.guard_spellcard(t)
            self.spell_card = HataSpellCard2(t, ms, self.beats, self.reimu)
        elif self.beats.ignite(0,0,2,3): #間奏後2番Aメロ
            self.reimu.guard_spellcard(t)
            self.spell_card = AbstractSpellCard(t, ms, self.beats, self.reimu)
        elif self.beats.ignite(0,0,1,5): #2サビ
            self.reimu.guard_spellcard(t)
            self.spell_card = HataSpellCard3(t, ms, self.beats, self.reimu)
        elif self.beats.ignite(0,0,3,7):
            self.reimu.guard_spellcard(t)
            self.spell_card = LastSpellCard(t, ms, self.beats, self.reimu)

        bullets = self.spell_card(t, ms, self.beats)
        self.print(f"bullets: {len(bullets)}")
        is_hit = False
        for bullet in bullets:
            bullet.draw(self.screen_playarea)
            if check_hit(self.reimu, bullet) and not is_hit and not controller_input.shot:
                is_hit = True

        self.reimu.update(t, controller_input, is_hit)
        self.reimu.draw(t, self.screen_playarea)

        self.screen_display.blit(self.screen_playarea, (BORDER_RECT.left, BORDER_RECT.top))
        self.screen_display.blit(self.screen_info, (INFO_RECT.left, INFO_RECT.top))
        pg.draw.rect(self.screen_display, cl.green, BORDER_RECT, 2)
    
    def print(self, txt: str) -> None:
        """INFOエリアにテキストを描画"""
        self.screen_info.blit(self.font.render(txt, True, cl.white), (0, self.fontoffset))
        self.fontoffset += self.fontsize

################ 弾幕とスペルカード ################
class AbstractBullet:
    def __init__(self, pos, radius) -> None:
        self.pos = pos; self.radius = radius
    def is_in_rect(self, RECT: pg.Rect) -> bool:
        return RECT.collidepoint(*self.pos)
    def draw(self, screen: pg.Surface) -> None: pass
class CircleBullet(AbstractBullet):
    def __init__(self, pos, radius, border, color) -> None:
        super().__init__(pos=pos, radius=radius)
        self.border = border; self.color = color
    def draw(self, screen):
        pg.draw.circle(screen, cl.white, self.pos, self.radius+self.border)
        pg.draw.circle(screen, self.color, self.pos, self.radius+self.border, self.border)
class SmallCircleBullet:
    # 弾数が2000近くになると60fpsを保てなくなるので継承は省く
    def __init__(self, pos, rgb):
        self.pos = pos; self.rgb = rgb
        self.radius = 2 # 外周の色部分には判定なし
    def draw(self, screen):
        pg.draw.circle(screen, (255, 255, 255), self.pos, 4)
        pg.draw.circle(screen, self.rgb, self.pos, 4, 2)
class MiddleCircleBullet(CircleBullet):
    def __init__(self, pos, color):
        super().__init__(pos=pos, radius=10, border=3, color=color)
class StraightBullet(MiddleCircleBullet):
    def __init__(self, pos, direction, speed=15, color=cl.red):
        super().__init__(pos=pos, color=color)
        self.direction = direction
        self.speed = speed
    def update(self):
        self.pos = self.pos + self.speed*self.direction

class AbstractSpellCard:
    def __init__(self, t, ms, beats, reimu):
        self.t = t
        self.ms = ms
        self.beats = beats
        self.reimu = reimu
    def __call__(self, t, ms, beats):
        return list()

def hatabullets(p):
    """パラメタpの畑写像から極小弾を返す"""
    zs = hatafast(p)
    retval = list()
    for i, z in enumerate(zs):
        rgb = hsv2rgb(i/len(zs), 1, 1)
        xy = z2xy(z)
        X = 2*np.array([-300+220*i for i in range(3)])
        Y = 2*np.array(range(-140, 300, 110))
        for x in X:
            for y in Y:
                _xy = (xy[0]+x, xy[1]+y)
                if ( PLAYAREA_RECT.left < _xy[0] < PLAYAREA_RECT.right and
                     PLAYAREA_RECT.top < _xy[1] < PLAYAREA_RECT.bottom 
                ):
                    retval.append(SmallCircleBullet(_xy, rgb))
    return retval
class HataSpellCard(AbstractSpellCard): #左右に小回りして避ける
    def __init__(self, t, ms, beats, reimu):
        super().__init__(t, ms, beats, reimu)
        self.params = [pr.はっぱや,pr.くるくる,pr.たちきの,pr.ひしがた]
    def __call__(self, t, ms, beats):
        t -= self.t
        T = 70 # 4beat/(190bpm/60sec)*60frame
        p = self.params[(t//T)%len(self.params)]
        q = self.params[(t//T+1)%len(self.params)]
        r = (1-t%T/T)*p+t%T/T*q
        return hatabullets(r)
class HataSpellCard2(HataSpellCard): #爆発するときに上が安置
    def __init__(self, t, ms, beats, reimu):
        super().__init__(t, ms, beats, reimu)
        self.params = [pr.さんかく,pr.おてがみ,pr.せきへん,pr.かこまれ]
class HataSpellCard3(HataSpellCard): #難しい、小刻みに右斜め下にもぐりこむ
    def __init__(self, t, ms, beats, reimu):
        super().__init__(t, ms, beats, reimu)
        self.params = [pr.ひびわれ,pr.みつびし,pr.ドラゴン,pr.くろすい]

def r() -> float:
    return 2*np.pi*random()

class OneSpellCard(AbstractSpellCard):
    def __init__(self, t, ms, beats, reimu):
        super().__init__(t, ms, beats, reimu)
        self.bullets = list()
        self.margin = 100
        self.color = cl.red
        self.lc = np.array([PLAYAREA_RECT.left+self.margin, PLAYAREA_RECT.top+self.margin])
        self.rc = np.array([PLAYAREA_RECT.right-self.margin, PLAYAREA_RECT.top+self.margin])
        self.way = 32

    def __call__(self, t, ms, beats):
        self.lc += (0, 1)
        self.rc += (0, 1)
        self.lcc = MiddleCircleBullet(self.lc, self.color)
        self.rcc = MiddleCircleBullet(self.rc, self.color)
        if beats[0] != beats[1]:
            for i in range(3):
                if beats.ignite(0,i,0,0):
                    r = 2*np.pi*random()
                    self.bullets += [StraightBullet(self.lc, circ((0,0),i*2*np.pi/self.way+r,1))for i in range(32)]
                if beats.ignite(2,i,0,0):
                    r = 2*np.pi*random()
                    self.bullets += [StraightBullet(self.rc, circ((0,0),i*2*np.pi/self.way+r,1))for i in range(32)]
            if beats.ignite(0,3,0,0):
                r = 2*np.pi*random()
                self.bullets += [StraightBullet(self.lc, circ((0,0),i*2*np.pi/self.way+r,1))for i in range(32)]
            if beats.ignite(1,3,0,0):
                r = 2*np.pi*random()
                self.bullets += [StraightBullet(self.lc, circ((0,0),i*2*np.pi/self.way+r,1))for i in range(32)]
            if beats.ignite(2,3,0,0):
                r = 2*np.pi*random()
                self.bullets += [StraightBullet(self.rc, circ((0,0),i*2*np.pi/self.way+r,1))for i in range(32)]

        for bullet in self.bullets:
            bullet.update()
        return self.bullets+[self.lcc, self.rcc]

class LastSpellCard(AbstractSpellCard):
    def __init__(self, t, ms, beats, reimu):
        super().__init__(t, ms, beats, reimu)
        self.center1 = np.array([PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.top*.5+PLAYAREA_RECT.bottom*.5])
        self.center2 = np.array([PLAYAREA_RECT.left*.5+PLAYAREA_RECT.right*.5, PLAYAREA_RECT.top-100])
        self.way = 18 
        self.earth_radius = 50
        self.cspeed = 0.0005
        self.const = 1
        self.gap = 70
        self.gain = self.cspeed/100
    def __call__(self, t, ms, beats):
        t -= self.t
        T = 60*10
        if T < t:
            self.center = self.center1
            self.cspeed += self.gain
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
        return self.lwind+self.rwind

class TitleStep:
    def __init__(self, screen: pg.Surface, clock: pg.time.Clock) -> None:
        self.screen_display = screen
        self.clock = clock
        self.screen_description = pg.Surface((PLAYAREA_RECT.width, PLAYAREA_RECT.height), pg.SRCALPHA)
        self.screen_playarea_color = cl.gray1
        self.description = pg.image.load("data/description4.png")
        self.bg = pg.image.load("data/bg.png")
        self.logo = pg.image.load("data/logo.png")
        self.offset = 0
        self.scroll_speed = 20

    def play(self, t: int, controller_input: Dict) -> None:
        if controller_input.down:
            self.offset -= self.scroll_speed
        elif controller_input.up:
            self.offset += self.scroll_speed
        if self.offset < -590: self.offset = -590
        if self.offset > 0: self.offset = 0
        global CONFIG

        if controller_input.left:
            CONFIG["bomb_stock"] -= 1
            se.ok00.play()
        elif controller_input.right:
            CONFIG["bomb_stock"] += 1
            se.ok00.play()
        if CONFIG["bomb_stock"] < 0: CONFIG["bomb_stock"] = 0

        self.screen_display.blit(self.bg, (0,0))
        self.screen_display.blit(self.logo, (DISPLAY_RECT.right-444, DISPLAY_RECT.bottom-390))
        self.screen_description.blit(self.description, (0,0))
        self.screen_display.blit(self.description, (BORDER_RECT.left+130, BORDER_RECT.top + self.offset))


class GameMainLoop:
    def __init__(self) -> None:
        pg.init()
        pg.display.set_caption("東方不動点")
        self.screen_display = pg.display.set_mode((DISPLAY_RECT.width, DISPLAY_RECT.height))
        self.controller = MyController()
        self.clock = pg.time.Clock()
        self.current_step = TitleStep
        se.cancel00 = pg.mixer.Sound("data/se_cancel00.wav")
        se.ok00 = pg.mixer.Sound("data/se_ok00.wav")
        se.slash = pg.mixer.Sound("data/se_slash.wav")
        se.pldead00 = pg.mixer.Sound("data/se_pldead00.wav")
        se.invalid = pg.mixer.Sound("data/se_invalid.wav")
        pg.mixer.music.load("data/Yours.wav")
        pg.mixer.music.play(loops=-1)

    def restart_game(self) -> None:
        self.current_step = GameStep
        pg.mixer.music.load("data/ENISHI.ogg")
        pg.mixer.music.stop()
        pg.mixer.music.rewind()
        pg.mixer.music.play(1, CONFIG["ms"]/1000)
        self.game_step = GameStep(self.screen_display, self.clock)
        self.t = 0
        se.ok00.play()
    
    def back_to_title(self) -> None:
        self.current_step = TitleStep
        pg.mixer.music.load("data/Yours.ogg")
        pg.mixer.music.stop()
        pg.mixer.music.rewind()
        pg.mixer.music.play(loops=-1)
        self.game_step = GameStep(self.screen_display, self.clock)
        se.cancel00.play()

    def main(self) -> None:
        self.game_step = GameStep(self.screen_display, self.clock)
        self.title_step = TitleStep(self.screen_display, self.clock)
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
            else:
                self.title_step.play(self.t, controller_input)
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
            self.t += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beat', type=str, default="0000")
    parser.add_argument('--bomb_stock', type=int, default=100)
    args = parser.parse_args()
    CONFIG["bomb_stock"] = args.bomb_stock
    CONFIG["beat"] = tuple(map(int, args.beat))
    CONFIG["ms"] = math.ceil(beat2ms(CONFIG["beat"]))
    print(CONFIG["beat"], CONFIG["ms"], ms2beat(CONFIG["ms"]), ms2beat(math.floor(CONFIG["ms"])), ms2beat(math.ceil(CONFIG["ms"])), ms2beat(round(CONFIG["ms"])))
    assert CONFIG["beat"] == ms2beat(CONFIG["ms"])
    game_main_loop = GameMainLoop()
    game_main_loop.main()