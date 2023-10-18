import pygame
import random
from pygame.rect import *


#pygame 초기화
pygame.init()
pygame.display.set_caption("rhythm game")

# pygame.mixer.music.load("세븐틴-02-Ready to love.mp3")
# pygame.mixer.music.play(0)

#======== 함수 ===============================
#키 이벤트 처리하기
def resultProcess(direction):
    global isColl, score, DrawResult, result_ticks, level, cycle
# 전역변수로 'isColl', 'score', 'DrawResult', 'result_ticks' 선언

# 충돌했을 때 방향과 충돌값이 모두 맞을 경우,
    if isColl and CollDirection.direction == direction:
        score += 10 # 점수를 10점 추가한다.
        CollDirection.y = -1 # 리셋한다.
        DrawResult = 1 # 웃는 얼굴 출력
        if score % 200 == 0:
            level += 1
            cycle -= 200

    else:
        DrawResult = 2 # 우는 얼굴 출력
    result_ticks = pygame.time.get_ticks() # result_ticks에 방향키 누른 시간을 부여

def eventProcess(): # 눌려진 키의 이벤트를 가져와서
    global isActive, score, level, start_time
    for event in pygame.event.get():
# 파이게임이 실행된다면
        if event.type == pygame.KEYDOWN:
# 키가 눌렸을 때 == pygame key down
            if event.key == pygame.K_ESCAPE:
                isActive = False
#이벤트 타입 파이게임 키 ESC라면 게임을 종료
                            
            if event.key == pygame.K_UP:  # 0 = 화살표 이미지 회전각도
                resultProcess(0)
            if event.key == pygame.K_LEFT:  # 1
                resultProcess(1)
            if event.key == pygame.K_DOWN:  # 2
                resultProcess(2)
            if event.key == pygame.K_RIGHT:  # 3
                resultProcess(3)

            if event.key == pygame.K_SPACE: # 스페이스바를 누르면 게임이 재시작
                score = 0 #스코어 0
                level = 1
                start_time = pygame.time.get_ticks()
                for direc in Directions:
                    direc.y = -1
###################################################################################
#방향 아이콘 클래스  # 이미지 한 장을 돌려서 사용
class Direction(object):
    def __init__(self):
        self.pos = None # 현재의 아이콘이 그리고 있는 좌표의 위치
        self.direction = 0
        self.image = pygame.image.load(f"direction.png")            # 이미지 불러오기
        self.image = pygame.transform.scale(self.image, (100, 100))   # 이미지 크기 설정
        self.rotated_image = pygame.transform.rotate(self.image, 0) # 이미지 회전
        self.y = -1  # y = -1일 때 이미지 그리지 않는다
         

    def rotate(self, direction=0):
        self.direction = direction
        self.rotated_image = pygame.transform.rotate(
            self.image, 90*self.direction)  # 0도, 90도, 180도, 270도 회전
        if (self.direction == 1):
            self.x = 50 
        if (self.direction == 0):
            self.x = 153  
        if (self.direction == 2):
            self.x = 256  
        if (self.direction == 3):
            self.x = 360 
    def draw(self):
        global cycle
        if self.y >= SCREEN_HEIGHT:
            self.y = -1   # 이미지가 스크린을 벗어나면 y = -1
            return True
        elif self.y == -1:
            return False
        else:
            self.y += 1+level   # 위의 경우가 전부 아닌 경우 y가 1씩 증가하며 이미지가 그려진다.
            self.pos = screen.blit(self.rotated_image, (self.x, self.y))
            return False

###################################################################################
#방향 아이콘 생성과 그리기
def drawIcon(): 
    global start_ticks, total_time, cycle

    remain_time = (pygame.time.get_ticks() - start_time) / 1000

    if total_time - remain_time <=0:
        return

    elapsed_time = (pygame.time.get_ticks() - start_ticks) # 지나간 시간 = 흘러가고 있는 시간 - 현재 시간
    if elapsed_time > cycle:  # 한 번에 400밀리초만 실행. 주기가 400밀리초
        start_ticks = pygame.time.get_ticks()
        for direc in Directions:  # 10개 리스트로 되어 있는 이미지 하나씩 가져오기
            if direc.y == -1: # direc이 리셋되어 있을 경우,
                direc.y = 0
                direc.rotate(direction=random.randint(0, 3))
                break

    for direc in Directions:  
        if direc.draw():        
            continue
###################################################################################
#타겟 영역 그리기와 충돌 확인하기
def draw_targetArea():
    global isColl, CollDirection
    isColl = False
    for direc in Directions:
        # Directions : direction의 10개의 리스트
        if direc.y == -1: # 방향 아이콘이 화면 밖으로 나갔을때 
            continue #스킵
        if direc.pos.colliderect(targetArea1): # 아이콘과 충돌 시
            isColl = True
            CollDirection = direc # CollDirection: 아이콘과 충돌 여부를 확인하는 변수
            pygame.draw.rect(screen, (80, 186, 169), targetArea1) # 초록색으로 채워줌
            break
        if direc.pos.colliderect(targetArea2): # 아이콘과 충돌 시
            isColl = True
            CollDirection = direc # CollDirection: 아이콘과 충돌 여부를 확인하는 변수
            pygame.draw.rect(screen, (80, 186, 169), targetArea2) # 초록색으로 채워줌
            break
        if direc.pos.colliderect(targetArea3): # 아이콘과 충돌 시
            isColl = True
            CollDirection = direc # CollDirection: 아이콘과 충돌 여부를 확인하는 변수
            pygame.draw.rect(screen, (80, 186, 169), targetArea3) # 초록색으로 채워줌
            break
        if direc.pos.colliderect(targetArea4): # 아이콘과 충돌 시
            isColl = True
            CollDirection = direc # CollDirection: 아이콘과 충돌 여부를 확인하는 변수
            pygame.draw.rect(screen, (80, 186, 169), targetArea4) # 초록색으로 채워줌
            break
    pygame.draw.rect(screen, (35, 35, 91), targetArea1, 5)  # 스크린에 파란색의 사각형을 5 두께로
    pygame.draw.rect(screen, (35, 35, 91), targetArea2, 5)
    pygame.draw.rect(screen, (35, 35, 91), targetArea3, 5)
    pygame.draw.rect(screen, (35, 35, 91), targetArea4, 5)
###################################################################################
#진행률 타임테이블
def timebar():
    global start_time, total_time
    remain_time = (pygame.time.get_ticks() - start_time) / 1000
    if total_time - remain_time > 0:
        time_bar = pygame.Rect(20, 50, 460, 50)
        real_time = (pygame.time.get_ticks() - start_time) / 1000 * 4.6 * 100 / total_time
        prog_time = pygame.Rect(23, 50, real_time, 50)
        prog_bar = pygame.draw.rect(screen, (221, 150, 168), prog_time)
        pygame.draw.rect(screen, (35, 35, 91), time_bar, 4)
    else:
        time_bar = pygame.Rect(20, 50, 460, 50)
        prog_time = pygame.Rect(23, 50, 455, 50)
        prog_bar = pygame.draw.rect(screen, (221, 150, 168), prog_time)
        pygame.draw.rect(screen, (35, 35, 91), time_bar, 4)
###################################################################################
#문자 넣기
def setText():
    global score, total_time, left_time
    remain_time = (pygame.time.get_ticks() - start_time) / 1000
    mFont = pygame.font.SysFont("굴림", 40)

    mtext = mFont.render(f'Score : {score}', True, 'black')   # 점수
    screen.blit(mtext, (10, 10, 0, 0))
    
    mtext = mFont.render(f'0%', True, 'black')
    screen.blit(mtext, (23, 100, 0, 0))

    mtext = mFont.render(f'100%', True, 'black')
    screen.blit(mtext, (420, 100, 0, 0))

    mtext = mFont.render(f'Level : {level}', True, 'black')
    screen.blit(mtext, (200, 10, 0, 0))


    if total_time - remain_time >= 0 :
        left_time = total_time - remain_time
    else :
        left_time == 0
    mFont = pygame.font.SysFont("굴림", 35)
    timer = mFont.render(f'Time : {str(int(left_time))}', True, 'black')
    screen.blit(timer, (380,10))

    if total_time - remain_time <= 0:
        mFont = pygame.font.SysFont("굴림", 60)
        mtext = mFont.render(f'TOTAL SCORE : {score}', True, 'red')   # 점수
        screen.blit(mtext, (60,250))
###################################################################################
#결과 이모티콘 그리기
def drawResult():
    global DrawResult, result_ticks, count, cycle  # 버튼 누르기 전 0값
    if result_ticks > 0:   # 알맞게 누르면 현재 시간을 부여받아 0보다 크다
        elapsed_time = (pygame.time.get_ticks() - result_ticks)  
        if elapsed_time > cycle:
            result_ticks = 0
            DrawResult = 0
        else:
            count += 1
    screen.blit(resultImg[DrawResult], resultImgRec)
##################################################################################
#========= 변수 =================================
isActive = True
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
score = 0
level = 1
isColl = False
CollDirection = 0
DrawResult, result_ticks = 0,0  # 0으로 초기화
count = 0
cycle = 1000  # 주기
start_ticks = pygame.time.get_ticks()  # 주기용
start_time = pygame.time.get_ticks()   # 시간 계산용

clock = pygame.time.Clock()
total_time = 50
left_time = total_time
# 스크린 생성
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) 
#배경
background = pygame.image.load("background.jpg")
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

#방향 아이콘
Directions = [Direction() for i in range(0, 10)]  # 이미지를 미리 10개 생성한 후 하나씩 떨어뜨리기
#타겟 박스
targetArea1 = Rect(60, 400, 85, 80)  #(x위치, y위치, 너비, 높이)
targetArea2 = Rect(160, 400, 85, 80)
targetArea3 = Rect(262, 400, 85, 80)
targetArea4 = Rect(362, 400, 85, 80)
#결과 이모티콘
resultFileNames = ["normal.png", "good.png", "bad.png"]
resultImg = []
for i, name in enumerate(resultFileNames):
    resultImg.append(pygame.image.load(name))
    resultImg[i] = pygame.transform.scale(resultImg[i], (100, 120))

#표시되는 이미지의 위치
resultImgRec = resultImg[0].get_rect()
resultImgRec.centerx = 250
resultImgRec.centery = 530

#========= 반복문 ===============================
while(isActive):
    screen.fill((255, 255, 255))  #스크린 배경색
    screen.blit(background, (0, 0))
    eventProcess()
    # Directions[0].y = 100
    # Directions[0].rotate(1)
    # Directions[0].draw()
    draw_targetArea()
    drawIcon()
    setText()
    drawResult()
    timebar()
    pygame.display.update()
    clock.tick(cycle)  # 1초에 800번 반복