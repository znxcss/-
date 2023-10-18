#타임테이블 추가

#진행률 타임테이블
def timebar():
    global start_time, total_time, remain_time
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
    global score, chance
    mFont = pygame.font.SysFont("굴림", 40)

    mtext = mFont.render(f'score : {score}', True, 'black')   # 점수
    screen.blit(mtext, (10, 10, 0, 0))
    
    mtext = mFont.render(f'0%', True, 'black')  # 진행률 0%
    screen.blit(mtext, (23, 100, 0, 0))

    mtext = mFont.render(f'100%', True, 'black') # 진행률 100%
    screen.blit(mtext, (420, 100, 0, 0))
###################################################################################
#========= 변수 =================================
isActive = True
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
score = 0
isColl = False
CollDirection = 0
DrawResult, result_ticks = 0,0  # 0으로 초기화
start_ticks = pygame.time.get_ticks()
start_time = pygame.time.get_ticks()

clock = pygame.time.Clock()
total_time = 
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # 스크린 생성
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
    clock.tick(800)  # 1초에 800번 반복