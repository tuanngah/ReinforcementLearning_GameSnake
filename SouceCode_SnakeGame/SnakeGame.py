import pygame
import random

pygame.init()

# Các thông số cơ bản của game
WIDTH, HEIGHT = 320, 240
GRID_SIZE = 20
WHITE = (255, 255, 255)
GREY = (50, 50, 50)
PINK = (255, 105, 180)

# Hướng di chuyển
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SnakeGame:
    '''
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game - Reinforcement Learning')
        self.clock = pygame.time.Clock()
        self.reset()  
    '''

    def __init__(self, render=False):
        self.render = render
        if self.render:
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Snake Game - Reinforcement Learning')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Khởi tạo lại trạng thái của trò chơi
        self.snake = [[WIDTH // 2, HEIGHT // 2]]
        self.direction = UP
        self.spawn_food()
        self.score = 0
        self.head = self.snake[0]
        return self._get_state()

    def spawn_food(self):
        # Sinh ra thức ăn tại vị trí ngẫu nhiên
        self.food = [random.randrange(0, WIDTH, GRID_SIZE),
                     random.randrange(0, HEIGHT, GRID_SIZE)]

    def play_step(self, action):
        # Xử lý hành động và tính toán trạng thái mới
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Cập nhật hướng di chuyển dựa trên hành động
        self._move(action)
        self.snake.insert(0, list(self.head))

        # Kiểm tra nếu ăn thức ăn
        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()

        # Kiểm tra nếu va chạm (thua cuộc)
        game_over = False
        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.calculate_score(), self._get_state()

        # Vẽ lại màn hình
        '''
        self.display.fill(WHITE)
        for segment in self.snake:
            pygame.draw.rect(self.display, GREY, pygame.Rect(segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.display, PINK, pygame.Rect(self.food[0], self.food[1], GRID_SIZE, GRID_SIZE))
        pygame.display.flip()
        '''

        if self.render:
            self.display.fill(WHITE)
            for segment in self.snake:
                pygame.draw.rect(self.display, GREY, pygame.Rect(segment[0], segment[1], GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.display, PINK, pygame.Rect(self.food[0], self.food[1], GRID_SIZE, GRID_SIZE))
            pygame.display.flip()

        # Tăng tốc độ rắn lên 20 FPS
        self.clock.tick(20)

        return reward, game_over, self.calculate_score(), self._get_state()

    def _move(self, action):
        # Hành động: [0, 1, 2] tương ứng với 'giữ nguyên', 'rẽ phải', 'rẽ trái'
        if action == 1:  # rẽ phải
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # rẽ trái
            self.direction = (self.direction - 1) % 4

        # Cập nhật vị trí đầu rắn
        x, y = self.head
        if self.direction == UP:
            y -= GRID_SIZE
        elif self.direction == RIGHT:
            x += GRID_SIZE
        elif self.direction == DOWN:
            y += GRID_SIZE
        elif self.direction == LEFT:
            x -= GRID_SIZE
        self.head = [x, y]

    def _is_collision(self):
        # Kiểm tra va chạm với tường hoặc chính thân
        if (self.head[0] < 0 or self.head[0] >= WIDTH or
            self.head[1] < 0 or self.head[1] >= HEIGHT):
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _get_state(self):
        # Tính trạng thái hiện tại (state) dưới dạng tọa độ
        head_x, head_y = self.head
        food_x, food_y = self.food

        # Tọa độ tương đối của thức ăn so với đầu rắn
        delta_x = (food_x - head_x) // GRID_SIZE
        delta_y = (food_y - head_y) // GRID_SIZE

        # Nguy cơ va chạm
        danger_straight = self._danger_straight()
        danger_left = self._danger_left()
        danger_right = self._danger_right()

        # Trả về state
        state = [
            delta_x,
            delta_y,
            self.direction,
            danger_straight,
            danger_left,
            danger_right
        ]

        return state

    def _danger_straight(self):
        # Kiểm tra nguy cơ va chạm khi tiếp tục đi thẳng
        x, y = self.head
        if self.direction == UP:
            y -= GRID_SIZE
        elif self.direction == RIGHT:
            x += GRID_SIZE
        elif self.direction == DOWN:
            y += GRID_SIZE
        elif self.direction == LEFT:
            x -= GRID_SIZE
        return int([x, y] in self.snake or x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT)

    def _danger_left(self):
        # Kiểm tra nguy cơ va chạm khi rẽ trái
        new_direction = (self.direction - 1) % 4
        return self._check_danger(new_direction)

    def _danger_right(self):
        # Kiểm tra nguy cơ va chạm khi rẽ phải
        new_direction = (self.direction + 1) % 4
        return self._check_danger(new_direction)

    def _check_danger(self, direction):
        # Kiểm tra nguy cơ va chạm dựa trên hướng
        x, y = self.head
        if direction == UP:
            y -= GRID_SIZE
        elif direction == RIGHT:
            x += GRID_SIZE
        elif direction == DOWN:
            y += GRID_SIZE
        elif direction == LEFT:
            x -= GRID_SIZE
        return int([x, y] in self.snake or x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT)

    def calculate_score(self):
        # Phương thức tính điểm cho agent
        return len(self.snake) - 1  # Điểm là số lượng các đoạn thân rắn đã có (trừ đầu rắn)

    def get_all_states(self):
        """Trả về tất cả các trạng thái có thể có của trò chơi"""
        states = []
        for x in range(0, WIDTH, GRID_SIZE):
            for y in range(0, HEIGHT, GRID_SIZE):
                for direction in [UP, RIGHT, DOWN, LEFT]:
                    for food_x in range(0, WIDTH, GRID_SIZE):
                        for food_y in range(0, HEIGHT, GRID_SIZE):
                            state = [
                                (food_x - x) // GRID_SIZE,  # delta_x
                                (food_y - y) // GRID_SIZE,  # delta_y
                                direction,
                                0, 0, 0  # Giả định rằng không có nguy cơ va chạm trong get_all_states
                            ]
                            states.append(state)
        return states

    def get_reward_and_next_state(self, state, action):
        """Lấy phần thưởng và trạng thái tiếp theo dựa trên hành động"""
        # Cập nhật vị trí của rắn theo hành động
        self._move(action)
        self.snake.insert(0, list(self.head))

        # Kiểm tra nếu ăn thức ăn
        if self.head == self.food:
            reward = 10  # Phần thưởng khi ăn thức ăn
            self.spawn_food()  # Sinh thức ăn mới
        else:
            self.snake.pop()
            reward = 0  # Không có phần thưởng

        # Kiểm tra va chạm
        if self._is_collision():
            reward = -10  # Phạt nếu va chạm

        # Lấy trạng thái tiếp theo
        next_state = self._get_state()

        return reward, next_state