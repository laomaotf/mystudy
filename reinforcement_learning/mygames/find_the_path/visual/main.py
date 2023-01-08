#!/usr/bin/env python
""" pygame.examples.aliens

Shows a mini game where you have to defend against aliens.

What does it show you about pygame?

* pg.sprite, the difference between Sprite and Group.
* dirty rectangle optimization for processing for speed.
* music with pg.mixer.music, including fadeout
* sound effects with pg.Sound
* event processing, keyboard handling, QUIT handling.
* a main loop frame limited with a game clock from pg.time.Clock
* fullscreen switching.


Controls
--------

* Left and right arrows to move.
* Space bar to shoot
* f key to toggle between fullscreen.

"""

import random
import os
import pickle
# import basic pygame modules
import pygame as pg
import numpy as np
import cv2
# see if we can load more than standard BMP
if not pg.image.get_extended():
    raise SystemExit("Sorry, extended image module required")


# game constants
MAX_SHOTS = 2  # most player bullets onscreen
ALIEN_ODDS = 22  # chances a new alien appears
BOMB_ODDS = 60  # chances a new bomb will drop
ALIEN_RELOAD = 12  # frames between new aliens
SCREENRECT = pg.Rect(0, 0, 640*2, 640)
SCORE = 0

main_dir = os.path.split(os.path.abspath(__file__))[0]


TRAJECTORY = {
            "size": (0,0),
            "apple": (0,0),
            "player": (0,0),
            "bomb": None
        }

def load_image(file):
    """loads an image, prepares it for play"""
    file = os.path.join(main_dir, "data", file)
    try:
        surface = pg.image.load(file)
    except pg.error:
        raise SystemExit('Could not load image "%s" %s' % (file, pg.get_error()))
    return surface.convert()


def load_sound(file):
    """because pygame can be be compiled without mixer."""
    if not pg.mixer:
        return None
    file = os.path.join(main_dir, "data", file)
    try:
        sound = pg.mixer.Sound(file)
        return sound
    except pg.error:
        print("Warning, unable to load, %s" % file)
    return None

def convert_system(position):
    global TRAJECTORY
    y,x = position
    W,H = SCREENRECT.width, SCREENRECT.height
    h,w = TRAJECTORY['size']
    if h > 0 and w > 0:
        x = int(x * W/w + 0.5*W/w)
        y = int(y * H/h + 0.5*H/h)
    return (y,x)  

# Each type of game object gets an init and an update function.
# The update function is called once per frame, and it is when each object should
# change it's current position and state.
#
# The Player object actually gets a "move" function instead of update,
# since it is passed extra information about the keyboard.


class Player(pg.sprite.Sprite):
    """Representing the player as a moon buggy type car."""

    bounce = 24
    gun_offset = -11
    images = []

    def __init__(self):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        #_,_,w,h= self.image.get_rect()
        #self.rect = pg.Rect(-1,SCREENRECT.h-1,w,h)
        
        self.rect = self.image.get_rect(center=(0,0))
        self.lastpos = (0,0)
        self.facing = -1

    def move(self, position):
        y,x = position
        y,x = convert_system((y,x))
        #if self.lastpos[0] < 0 or self.lastpos[1] < 0:
        #    self.lastpos = (y,x)
        lasty,lastx = self.lastpos
        direction = 0
        if x - lastx >  0:
            direction = 1
        elif x - lastx < 0:
            direction = -1
        
        if direction:
            self.facing = direction
        self.rect.move_ip(x-lastx,y-lasty)
        self.rect = self.rect.clamp(SCREENRECT)
        if direction < 0:
            self.image = self.images[0]
        elif direction > 0:
            self.image = self.images[1]
        #self.rect.top = self.origtop - (self.rect.left // self.bounce % 2)
        self.lastpos = (y,x)

    def gunpos(self):
        pos = self.facing * self.gun_offset + self.rect.centerx
        return pos, self.rect.top


class Alien(pg.sprite.Sprite):
    """An alien space ship. That slowly moves down the screen."""

    speed = 13
    animcycle = 12
    images = []

    def __init__(self):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.facing = random.choice((-1, 1)) * Alien.speed
        self.frame = 0
        if self.facing < 0:
            self.rect.right = SCREENRECT.right

    def update(self):
        self.rect.move_ip(self.facing, 0)
        if not SCREENRECT.contains(self.rect):
            self.facing = -self.facing
            #self.rect.top = self.rect.bottom + 1
            self.rect = self.rect.clamp(SCREENRECT)
        self.frame = self.frame + 1
        self.image = self.images[self.frame // self.animcycle % 3]


class Explosion(pg.sprite.Sprite):
    """An explosion. Hopefully the Alien and not the player!"""

    defaultlife = 12
    animcycle = 3
    images = []

    def __init__(self, actor):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect(center=actor.rect.center)
        self.life = self.defaultlife

    def update(self):
        """called every time around the game loop.

        Show the explosion surface for 'defaultlife'.
        Every game tick(update), we decrease the 'life'.

        Also we animate the explosion.
        """
        self.life = self.life - 1
        self.image = self.images[self.life // self.animcycle % 2]
        if self.life <= 0:
            self.kill()


class Apple(pg.sprite.Sprite):
    """a bullet the Player sprite fires."""

    images = []

    def __init__(self, pos):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        y,x = convert_system(pos)
        self.rect = self.image.get_rect(center=(x,y))
        self.lastpos = (y,x)

    def update(self):
        """called every time around the game loop.

        Every tick we move the shot upwards.
        """
        global TRAJECTORY, SCORE
        appley, applex = TRAJECTORY['apple']
        y,x = convert_system((appley,applex))
        dy,dx = y - self.lastpos[0], x - self.lastpos[1] 
        if dx != 0 or dy != 0:
            SCORE += 1 
        self.rect.move_ip(dx,dy)
        self.lastpos = (y,x)
       


class Bomb(pg.sprite.Sprite):
    """A bomb the aliens drop."""

    images = []

    def __init__(self, index):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        _,_,w,h= self.image.get_rect()
        self.index = index
        _,x = convert_system((0,index))
        self.rect = self.image.get_rect(center=(x,-h))
        self.rect_hidden = self.image.get_rect(center=(x,-h))
        self.lasty = -h

    def update(self):
        """called every time around the game loop.

        Every frame we move the sprite 'rect' down.
        When it reaches the bottom we:

        - make an explosion.
        - remove the Bomb.
        """
        global TRAJECTORY
        bomb = TRAJECTORY['bomb']
        if 0:
            m0,m1 = bomb.min(), bomb.max()
            R = 15
            vis = np.clip((bomb - m0) * 255 / (m1-m0),0,255).astype(np.uint8)
            vis = cv2.resize(vis, (0,0),fx=R,fy=R)
            vis = cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)
            y,x = TRAJECTORY['apple']
            cv2.circle(vis,(x*10,y*10),3,(0,255,255),3)
            y,x = TRAJECTORY['player']
            cv2.circle(vis,(x*10,y*10),3,(0,255,0),1)
            
            cv2.imshow("RL-small",vis)
            cv2.waitKey(10)
        
        if max(bomb[:,self.index]) == 0:
            self.rect = self.rect_hidden
            return
        index_y = np.argmax(bomb[:,self.index])
        y,_ = convert_system((index_y,0))
        self.rect.move_ip(0, y - self.lasty)
        self.lasty = y
        
        playery,playerx = TRAJECTORY['player']
        if self.index == playerx and index_y == playery: 
            Explosion(self)
            self.rect = self.rect_hidden
            self.lasty = self.rect.y

class Score(pg.sprite.Sprite):
    """to keep track of the score."""

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.font = pg.font.Font(None, 20)
        self.font.set_italic(1)
        self.color = pg.Color("white")
        self.lastscore = -1
        self.update()
        self.rect = self.image.get_rect().move(10, SCREENRECT.height - 100)

    def update(self):
        """We only update the score in update() when it has changed."""
        if SCORE != self.lastscore:
            self.lastscore = SCORE
            msg = "Score: %d" % SCORE
            self.image = self.font.render(msg, 0, self.color)


def main(winstyle=0):
    # Initialize pygame
    if pg.get_sdl_version()[0] == 2:
        pg.mixer.pre_init(44100, 32, 2, 1024)
    pg.init()
    if pg.mixer and not pg.mixer.get_init():
        print("Warning, no sound")
        pg.mixer = None

    fullscreen = False
    # Set the display mode
    winstyle = 0  # |FULLSCREEN
    bestdepth = pg.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pg.display.set_mode(SCREENRECT.size, winstyle, bestdepth)

    # Load images, assign to sprite classes
    # (do this before the classes are used, after screen setup)
    img = pg.transform.scale(load_image("player1.gif"),(32,32))
    Player.images = [img, pg.transform.flip(img, 1, 0)]
    img = load_image("explosion1.gif")
    Explosion.images = [img, pg.transform.flip(img, 1, 1)]
    Alien.images = [pg.transform.scale(load_image(im),(32,32)) for im in ("alien1.gif", "alien2.gif", "alien3.gif")]
    Bomb.images = [pg.transform.scale(pg.transform.flip(load_image("shot.gif"),0,1),(32//2,32//2))]
    Apple.images = [pg.transform.scale(load_image("apple.gif"),(32//2,32//2))]

    # decorate the game window
    icon = pg.transform.scale(Alien.images[0], (32, 32))
    pg.display.set_icon(icon)
    pg.display.set_caption("RL")
    pg.mouse.set_visible(0)

    # create the background, tile the bgd image
    bgdtile = load_image("background.gif")
    bgdtile = pg.transform.scale(bgdtile,SCREENRECT.size)
    background = pg.Surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, bgdtile.get_width()):
        background.blit(bgdtile, (x, 0))
    screen.blit(background, (0, 0))
    pg.display.flip()

    # load the sound effects
    boom_sound = load_sound("boom.wav")
    shoot_sound = load_sound("car_door.wav")
    if pg.mixer:
        music = os.path.join(main_dir, "data", "house_lo.wav")
        pg.mixer.music.load(music)
        pg.mixer.music.play(-1)

    # Initialize Game Groups
    aliens = pg.sprite.Group()
    apples = pg.sprite.Group()
    bombs = pg.sprite.Group()
    all = pg.sprite.RenderUpdates()
    lastalien = pg.sprite.GroupSingle()

    # assign default groups to each sprite class
    Player.containers = all
    Alien.containers = aliens, all, lastalien
    Apple.containers = apples, all
    Bomb.containers = bombs, all
    Explosion.containers = all
    Score.containers = all

    # Create Some Starting Values
    global score
    alienreload = ALIEN_RELOAD
    clock = pg.time.Clock()

    # initialize our starting sprites
    global SCORE
    player = Player()
    Alien()  # note, this 'lives' because it goes into a sprite group
    if pg.font:
        all.add(Score())

        
    # Run our main loop whilst the player is alive.
    epoch = 0
    with open(os.path.join(main_dir,"trajectory.pkl"),'rb') as f:
        trajectory_full = pickle.load(f)
    
    global TRAJECTORY
    TRAJECTORY['size'] = trajectory_full['size']
    TRAJECTORY['apple'] = trajectory_full['apple'][0]
    TRAJECTORY['player'] = trajectory_full['player'][0]
    TRAJECTORY['bomb'] = trajectory_full['bomb'][0]
            
    for x in range(trajectory_full['size'][1]):
        Bomb(x)
        
    Apple(TRAJECTORY['apple'])
        
    while player.alive():
        #print(epoch, len(trajectory_full['player']))
        if epoch < len(trajectory_full['player']):
            TRAJECTORY['size'] = trajectory_full['size']
            TRAJECTORY['apple'] = trajectory_full['apple'][epoch]
            TRAJECTORY['player'] = trajectory_full['player'][epoch]
            TRAJECTORY['bomb'] = trajectory_full['bomb'][epoch]
        else:
            break
        epoch += 1
        # get input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    if not fullscreen:
                        print("Changing to FULLSCREEN")
                        screen_backup = screen.copy()
                        screen = pg.display.set_mode(
                            SCREENRECT.size, winstyle | pg.FULLSCREEN, bestdepth
                        )
                        screen.blit(screen_backup, (0, 0))
                    else:
                        print("Changing to windowed mode")
                        screen_backup = screen.copy()
                        screen = pg.display.set_mode(
                            SCREENRECT.size, winstyle, bestdepth
                        )
                        screen.blit(screen_backup, (0, 0))
                    pg.display.flip()
                    fullscreen = not fullscreen

        keystate = pg.key.get_pressed()

        # clear/erase the last drawn sprites
        all.clear(screen, background)

        # update all the sprites
        all.update()

        # handle player input
        #direction = keystate[pg.K_RIGHT] - keystate[pg.K_LEFT]
        player.move(TRAJECTORY['player'])
        # Create new alien
        if alienreload:
            alienreload = alienreload - 1
        elif not int(random.random() * ALIEN_ODDS):
            Alien()
            alienreload = ALIEN_RELOAD



        # Detect collisions between aliens and players.
        # for alien in pg.sprite.spritecollide(player, aliens, 1):
        #     if pg.mixer:
        #         boom_sound.play()
        #     Explosion(alien)
        #     Explosion(player)
        #     SCORE = SCORE + 1
        #     player.kill()

        # See if shots hit the aliens.
        # for alien in pg.sprite.groupcollide(aliens, shots, 1, 1).keys():
        #     if pg.mixer:
        #         boom_sound.play()
        #     Explosion(alien)
        #     SCORE = SCORE + 1

        # See if alien boms hit the player.
        # for bomb in pg.sprite.spritecollide(player, bombs, 1):
        #     if pg.mixer:
        #         boom_sound.play()
        #     Explosion(player)
        #     Explosion(bomb)
        #     player.kill()

        # draw the scene
        dirty = all.draw(screen)
        pg.display.update(dirty)

        # cap the framerate at 40fps. Also called 40HZ or 40 times per second.
        clock.tick(10)

    if pg.mixer:
        pg.mixer.music.fadeout(1000)
    pg.time.wait(1000)


# call the "main" function if running this script
if __name__ == "__main__":
    pg.time.wait(10*1000)
    main()
    pg.quit()
