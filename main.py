# %%
import pygame
from pygame.locals import *
from typing import NamedTuple
import sys
import colorsys
from addict import Dict
from itertools import count
import numpy as np
import math

DISPLAY_RECT = pygame.Rect(0, 0, 1280, 960)
BORDER_RECT = pygame.Rect(62, 30, 834-62, 930-30)
PLAYABLE_RECT = pygame.Rect(0, 0, 2*386, 2*450)
INFO_RECT = pygame.Rect(834+30, 30, 1280-834-2*30, 500)

class MyDict(dict):
    def at(self, k):
        return self[list(self.keys())[k]]

params = MyDict(
    はっぱや=np.array([+0.7+0.2j, +0.0 +0.0j, +0.0+0.0j, +0.65+0.0j]),
    さんかく=np.array([+0.0+0.0j, +0.45+0.5j, +0.0+0.0j, +0.45-0.5j]), 
    くるくる=np.array([0.4614+0.4614j,0,0.622-0.196j,0]),
    たちきの=np.array([0,0.3+0.3j,0,41/50]),
    ひしがた=np.array([0,0.5+0.5j,0,-0.5+0.5j]),
    かいそう=np.array([0.4614+0.4614j,0,0,0.2896-0.585j]),
)

def fast(p, z=np.array([[1, 1, 1]])):
    """畑写像を高速に計算"""
    A0 = [np.array([
        [ p[0].real+p[1].real, p[0].imag+p[1].imag, 0],    
        [-p[0].imag+p[1].imag, p[0].real-p[1].real, 0],
        [0                  ,  0                  , 1]
    ]), np.array([
        [ p[2].real+p[3].real  ,  p[2].imag+p[3].imag, 0],
        [-p[2].imag+p[3].imag  ,  p[2].real-p[3].real, 0],
        [-p[2].real-p[3].real+1, -p[2].imag-p[3].imag, 1]
    ])]
    A1 = [A0[i&1]@A0[(i>>1)&1] for i in range(4)]
    A = np.hstack([A1[i&3]@A1[(i>>2)&3] for i in range(16)])
    z = (z@A).reshape(-1, 3)
    z = (z@A).reshape(-1, 3)
    return z[:, :-1]

def z2xy(z):
    """画面に収まる位置に座標をスケール"""
    scale = min(PLAYABLE_RECT.width, PLAYABLE_RECT.height)/2
    x = z[0]*scale+PLAYABLE_RECT.width/2
    y = -z[1]*scale+PLAYABLE_RECT.height/2
    return (int(x), int(y))

def hsv2rgb(h,s,v):
    return tuple(round(i*255) for i in colorsys.hsv_to_rgb(h,s,v))

def hatafast(p):
    """パラメタpの畑写像から極小弾を返す"""
    zs = fast(p)
    retval = list()
    for i, z in enumerate(zs):
        rgb = hsv2rgb(i/len(zs), 1, 1)
        xy = z2xy(z)
        X = 2*np.array([-220, 0, 220]) 
        Y = 2*np.array(list(range(-140, 300, 110)))
        for x in X:
            for y in Y:
                _xy = (xy[0]+x, xy[1]+y)
                if ( PLAYABLE_RECT.left < _xy[0] < PLAYABLE_RECT.right and
                     PLAYABLE_RECT.top < _xy[1] < PLAYABLE_RECT.bottom 
                ):
                    retval.append(SmallCircleBullet(_xy, rgb))
    return retval

class SmallCircleBullet:
    def __init__(self, pos, rgb):
        self.pos = pos; self.rgb = rgb
        self.radius = 4
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.pos, 4)
        pygame.draw.circle(screen, self.rgb, self.pos, 4, 2)


class Status:
    KEY = ["left", "right", "up", "down", "slow", "shot", "bomb"]
    def __init__(self):
        self.now= False
        self.last = [-60*60*60, -60*60*60] # 1時間前から定常状態

class Reimu:
    def __init__(self):
        self.pl00 = pygame.image.load("data/w2x_pl00.png")
        self.pl10 = pygame.image.load("data/w2x_pl10.png")
        self.pos = np.array((390, 420))
        self.option = pygame.Surface((32, 32), pygame.SRCALPHA)
        self.option.blit(self.pl00, (0, 0), (3*32*2, 3*48*2, 16*2, 16*2))
        self.sloweffect = pygame.Surface((128, 128), pygame.SRCALPHA)
        self.sloweffect.blit(pygame.image.load("data/w2x_eff_sloweffect.png"), (0, 0), (0,0, 128, 128))
        self.eff_charge = pygame.Surface((64, 64), pygame.SRCALPHA)
        self.eff_charge.blit(pygame.image.load("data/w2x_eff_charge.png"), (0, 0), (0,0, 128, 128))
        self.bomb_stock = 100
        self.bomb_invincible = False
        self.bomb_invincible_time = 2*60 # ボム無敵時間
        self.bomb_lasttime = float("inf")
        self.hit_invincible = False
        self.hit_invincible_time = 1*60 # 被弾無敵時間
        self.hit_lasttime = float("inf")
        self.se_slash = pygame.mixer.Sound("data/se_slash.wav")
        self.se_pldead00 = pygame.mixer.Sound("data/se_pldead00.wav")
        self.radius = 3

        self.s = Dict({k:Status() for k in Status.KEY})

    def update(self, t, controller_input, is_hit):
        for k in Status.KEY:
            if self.s[k].now^controller_input[k]:
                self.s[k].last[controller_input[k]] = t
                self.s[k].now = controller_input[k]
        speed = 2*[4.0, 2.0][controller_input.slow]
        if [controller_input.left, controller_input.right, controller_input.up, controller_input.down].count(True) == 2:
            speed /= math.sqrt(2)

        if controller_input.left:
            self.pos[0] -= speed
        elif controller_input.right:
            self.pos[0] += speed

        if controller_input.up:
            self.pos[1] -= speed
        elif controller_input.down:
            self.pos[1] += speed
        
        # 画面外脱出防止
        mergin = 40
        if self.pos[0] < PLAYABLE_RECT.left+mergin: self.pos[0] = PLAYABLE_RECT.left+mergin
        if self.pos[0] > PLAYABLE_RECT.right-mergin: self.pos[0] = PLAYABLE_RECT.right-mergin
        if self.pos[1] < PLAYABLE_RECT.top+mergin: self.pos[1] = PLAYABLE_RECT.top+mergin
        if self.pos[1] > PLAYABLE_RECT.bottom-mergin: self.pos[1] = PLAYABLE_RECT.bottom-mergin

        # ボムの処理
        if ( self.s["bomb"].now and not self.bomb_invincible 
        and not self.hit_invincible and 0 < self.bomb_stock ):
            self.bomb_invincible = True
            self.bomb_stock -= 1
            self.bomb_lasttime = t
            self.se_slash.play()
        if 0 < t-self.bomb_lasttime-self.bomb_invincible_time:
            self.bomb_invincible = False

        if is_hit and not self.bomb_invincible and not self.hit_invincible:
            self.hit_invincible = True
            self.bomb_stock -= 2
            self.hit_lasttime = t
            self.se_pldead00.play()

        if 0 < t-self.hit_lasttime-self.hit_invincible_time:
            self.hit_invincible = False

        if self.bomb_stock < 0:
            pass # TODO

    def draw(self, t, screen):
        if self.s.left.now: reimu_offset = 1
        elif self.s.right.now: reimu_offset = 2
        else: reimu_offset = 0

        screen.blit(
            self.pl10 if self.bomb_invincible or self.hit_invincible else self.pl00,
            self.pos-(16*2, 24*2), (0+32*(t//8%4)*2, reimu_offset*48*2, 32*2, 48*2))
        if self.bomb_invincible:
            bomb_radius = self.bomb_invincible_time*(1-(t-self.bomb_lasttime)/self.bomb_invincible_time)
            pygame.draw.circle(screen, (0, 255, 255), self.pos, bomb_radius, 2)
        elif self.hit_invincible:
            hit_radius = self.hit_invincible_time*(1-(t-self.hit_lasttime)/self.hit_invincible_time)
            pygame.draw.circle(screen, (0, 0, 255), self.pos, hit_radius, 2)


        angle_option = -t*6%360
        d = 2*calc_d(angle_option)
        rot_option = pygame.transform.rotate(self.option, angle_option)

        angle_sloweffect = t*3%360
        d2 = 2*calc_d(angle_sloweffect)
        rot_sloweffect = pygame.transform.rotate(self.sloweffect, angle_sloweffect)
        rot_sloweffect2 = pygame.transform.rotate(self.sloweffect, -angle_sloweffect)

        option_pos_lower = np.array([[+16, 32], [-16, 32], [+38, 16], [-38, 16]])*2
        option_pos_upper = np.array([[+8, -30], [-8, -30], [+24, -20], [-24, -20]])*2
        option_move = 4
        dt = (t-self.s.slow.last[self.s.slow.now])/option_move
        if 0 <= dt < 1:
            if self.s.slow.now:
                option_pos = dt*option_pos_upper+(1-dt)*option_pos_lower
            else:
                option_pos = dt*option_pos_lower+(1-dt)*option_pos_upper
        else:
            if self.s.slow.now:
                option_pos = option_pos_upper
            else:
                option_pos = option_pos_lower

        for o in option_pos:
            screen.blit(self.eff_charge, self.pos-(32,32)+o, (0, 0, *self.eff_charge.get_size()))
            screen.blit(rot_option, self.pos-(8*d, 8*d)+o, (0, 0, *rot_option.get_size()))

        if self.s.slow.now:
            screen.blit(rot_sloweffect, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))
            screen.blit(rot_sloweffect2, self.pos-(32*d2, 32*d2), (0, 0, *rot_sloweffect.get_size()))
#        pygame.draw.circle(screen, (0, 0, 255), self.pos, 2)

def calc_d(angle):
    return math.cos(angle%90/360*2*math.pi)+math.sin(angle%90/360*2*math.pi)

def check_input(joystick):
    axis0 = joystick.get_axis(0)
    axis1 = joystick.get_axis(1)
    return Dict(
        left =axis0 < -0.5,
        right=axis0 > +0.5,
        up   =axis1 < -0.5,
        down =axis1 > +0.5,
        slow =bool(joystick.get_button(7)),
        bomb =bool(joystick.get_button(3)),
        shot =bool(joystick.get_button(2)),
    )

class MyController:
    def __init__(self):
        pass
    def check_input(self):
        keys = pygame.key.get_pressed()
        return Dict(
            left =keys[pygame.K_LEFT],
            right=keys[pygame.K_RIGHT],
            up   =keys[pygame.K_UP],
            down =keys[pygame.K_DOWN],
            slow =keys[pygame.K_LSHIFT],
            bomb =keys[pygame.K_x],
            shot =keys[pygame.K_z],
            esc  =keys[pygame.K_ESCAPE],
            retry=keys[pygame.K_r],
        )


class GameStep:
    def __init__(self, screen, clock) -> None:
        self.screen = screen
        self.clock = clock
        self.screen2 = pygame.Surface((PLAYABLE_RECT.width, PLAYABLE_RECT.height))
        self.screen3 = pygame.Surface((INFO_RECT.width, INFO_RECT.height))
        #font = pygame.font.SysFont(None, 40)
        #pygame.font.get_fonts()で確認、ttfのフルパス指定
        self.font = pygame.font.SysFont('yumincho', 40)
        self.bg = pygame.image.load("data/bg.png")
        self.logo = pygame.image.load("data/logo.png")
        self.reimu = Reimu()

        self.screen.blit(self.bg, (0,0))
        self.screen.blit(
            self.logo,
            (DISPLAY_RECT.right-444, DISPLAY_RECT.bottom-390)
        )
#  4/(190/60)*60
    def play(self, t, controller_input) -> None:
        T = 76
        #self.screen.fill((0,0,0))
        self.screen2.fill((20,20,20))
        self.screen3.fill((24,49,125))
        # テキスト描画処理
        self.screen3.blit(
            self.font.render(f"Bomb: ☆x{self.reimu.bomb_stock}", True, (255, 255, 255)),
            (0, 0)
        )
        self.screen3.blit(
            self.font.render(f"fps:{self.clock.get_fps():.2f}", True, (255, 255, 255)),
            (0, 50)
        )
        self.screen3.blit(
            self.font.render(f"hit={self.reimu.hit_invincible, t-self.reimu.hit_lasttime-self.reimu.hit_invincible_time}", True, (255, 255, 255)),
            (0, 100)
        )
        self.screen3.blit(
            self.font.render(f"bomb={self.reimu.bomb_invincible, t-self.reimu.bomb_lasttime-self.reimu.bomb_invincible_time}", True, (255, 255, 255)),
            (0, 150)
        )

        p = params.at((t//T)%len(params))
        q = params.at((t//T+1)%len(params))
        r = (1-t%T/T)*p+t%T/T*q

        is_hit = False
        for bullet in hatafast(r):
            bullet.draw(self.screen2)
            if not is_hit and ( (self.reimu.pos[0]-bullet.pos[0])**2 + (self.reimu.pos[1]-bullet.pos[1])**2
                < (self.reimu.radius+bullet.radius)**2 ):
                is_hit = True

        self.reimu.update(t, controller_input, is_hit)
        self.reimu.draw(t, self.screen2)

        self.screen.blit( self.screen2, (BORDER_RECT.left, BORDER_RECT.top))
        self.screen.blit(self.screen3, (INFO_RECT.left, INFO_RECT.top))
        pygame.draw.rect(self.screen, (0,255,0), BORDER_RECT, 2)



def main():
    pygame.init()
    pygame.display.set_caption("東方不動点")
    pygame.mixer.music.load("data/ENISHI.wav")
    pygame.mixer.music.play()
    screen = pygame.display.set_mode((DISPLAY_RECT.width, DISPLAY_RECT.height))
    controller = MyController()
    clock = pygame.time.Clock()
    game_step = GameStep(screen, clock)
    t = 0
    while True:
        controller_input = controller.check_input()
        game_step.play(t, controller_input)
        clock.tick(60)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
        if controller_input.esc:
            t = 0
        if controller_input.retry:
            pygame.quit(); sys.exit()
        t += 1



if __name__ == "__main__":
    main()