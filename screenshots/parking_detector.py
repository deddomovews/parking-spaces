import cv2
import numpy as np
import os
import urllib.request
import json
import time
import ftplib


class UcozFTPUploader:
    def __init__(self):
        # НАСТРОЙТЕ ЭТИ ПАРАМЕТРЫ ПОД ВАШ САЙТ uCoz
        self.ftp_config = {
            'host': 'parkofka.ucoz.org',  # Ваш домен uCoz
            'username': 'fparkofka',  # FTP логин
            'password': 'qwerty7',  # FTP пароль
            'remote_path': '/parking_status.json'  # Путь на сервере
        }
        self.enabled = True  # Включить/выключить загрузку

    def upload(self, json_data):
        """Загружает JSON файл на uCoz через FTP"""
        if not self.enabled:
            return False

        try:
            # Преобразуем данные в строку JSON
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)

            # Создаем временный файл
            temp_file = 'temp_parking_status.json'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

            # Подключаемся к FTP
            print(f"Подключение к FTP {self.ftp_config['host']}...")
            ftp = ftplib.FTP(self.ftp_config['host'], timeout=10)
            ftp.login(self.ftp_config['username'], self.ftp_config['password'])

            # Устанавливаем пассивный режим (часто требуется)
            ftp.set_pasv(True)

            # Загружаем файл
            with open(temp_file, 'rb') as f:
                ftp.storbinary(f'STOR {self.ftp_config["remote_path"]}', f)

            # Закрываем соединение
            ftp.quit()

            # Удаляем временный файл
            os.remove(temp_file)

            print(f"✓ Файл успешно загружен на uCoz: {time.strftime('%H:%M:%S')}")
            return True

        except ftplib.all_errors as e:
            print(f"✗ Ошибка FTP: {e}")
            return False
        except Exception as e:
            print(f"✗ Общая ошибка: {e}")
            return False


class ParkingDetector:
    def __init__(self):
        self.net = None
        self.classes = []
        self.output_layers = []
        self.parking_spots = []
        self.parking_file = "parking_spots.json"
        self.status_file = "parking_status.json"
        self.drawing = False
        self.current_spot = []
        self.current_mouse_pos = (0, 0)
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.last_status_print = 0

        # Инициализация FTP загрузчика
        self.uploader = UcozFTPUploader()
        self.last_upload_time = 0
        self.upload_interval = 5  # Интервал загрузки в секундах
        self.upload_enabled = True  # Включить автоматическую загрузку

    def download_yolo_files(self):
        files = {
            'yolov4.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
            'yolov4.cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }

        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"File {filename} downloaded!")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    return False
            else:
                print(f"File {filename} already exists")
        return True

    def load_model(self):
        print("Loading YOLOv4 model...")
        try:
            self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names = self.net.getLayerNames()
            try:
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            print(f"Model loaded! Classes: {len(self.classes)}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_parking_spots(self):
        if os.path.exists(self.parking_file):
            try:
                with open(self.parking_file, 'r') as f:
                    self.parking_spots = json.load(f)
                print(f"Loaded {len(self.parking_spots)} парковочные места")
            except:
                self.parking_spots = []
        else:
            self.parking_spots = []

    def save_parking_spots(self):
        with open(self.parking_file, 'w') as f:
            json.dump(self.parking_spots, f, indent=2)
        print(f"Saved {len(self.parking_spots)} парковочные места")

    def save_current_status(self, occupied_count, total_spots):
        """Сохраняет текущий статус парковки локально и на uCoz"""
        # Создаем данные в нужном формате
        status_data = []

        for spot in self.parking_spots:
            status_data.append({
                'id': spot['id'],
                'occupied': spot['occupied']
            })

        # Сохраняем локально
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)

            # Для отладки - выводим статус сохранения
            if self.frame_count % 100 == 0:
                print(f"Локальный статус сохранен: {occupied_count}/{total_spots} занято")

        except Exception as e:
            print(f"Ошибка сохранения локального файла: {e}")

        # Загружаем на uCoz через FTP (каждые upload_interval секунд)
        if self.upload_enabled and status_data:
            current_time = time.time()
            if current_time - self.last_upload_time > self.upload_interval:
                if self.uploader.upload(status_data):
                    self.last_upload_time = current_time

    def mouse_callback(self, event, x, y, flags, param):
        self.current_mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                # Начинаем рисовать новое парковочное место
                self.drawing = True
                self.current_spot = [(x, y)]
                print("Начало рисования парковочного места. Кликните еще 3 точки.")
            else:
                # Добавляем точку к текущему парковочному месту
                if len(self.current_spot) < 4:
                    self.current_spot.append((x, y))
                    print(f"Добавлена точка {len(self.current_spot)}: ({x}, {y})")

                    # Если набрали 4 точки, завершаем рисование
                    if len(self.current_spot) == 4:
                        self.finish_parking_spot()

    def finish_parking_spot(self):
        """Завершает создание парковочного места по 4 точкам"""
        if len(self.current_spot) == 4:
            # Проверяем, что площадь достаточно большая
            points = np.array(self.current_spot, dtype=np.int32)
            area = cv2.contourArea(points)

            if area > 500:  # Минимальная площадь
                spot_id = len(self.parking_spots) + 1
                self.parking_spots.append({
                    'id': spot_id,
                    'points': self.current_spot.copy(),
                    'occupied': False
                })
                print(f"Добавлено парковочное место {spot_id} с 4 точками")

                # Сохраняем споты и обновляем статус
                self.save_parking_spots()
                if self.parking_spots:
                    self.save_current_status(0, len(self.parking_spots))

            else:
                print("Парковочное место слишком маленькое!")

            self.drawing = False
            self.current_spot = []

    def detect_vehicles(self, frame, confidence_threshold=0.4):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.output_layers)

        vehicles = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold and class_id in [2, 5, 7,
                                                                      3]:  # 2: car, 5: bus, 7: truck, 3: motorcycle
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    vehicles.append({
                        'class': self.classes[class_id],
                        'confidence': confidence,
                        'bbox': [x, y, w, h]
                    })

        return vehicles

    def check_parking_occupancy(self, vehicles):
        for spot in self.parking_spots:
            spot['occupied'] = False

            if 'points' in spot:
                # Для полигонального парковочного места
                points = np.array(spot['points'], dtype=np.int32)
                spot_bbox = cv2.boundingRect(points)
                x1, y1, w, h = spot_bbox
                x2, y2 = x1 + w, y1 + h
                spot_area = w * h
            else:
                # Для прямоугольного парковочного места (обратная совместимость)
                x1, y1, x2, y2 = spot['coords']
                spot_area = (x2 - x1) * (y2 - y1)

            for vehicle in vehicles:
                vx, vy, vw, vh = vehicle['bbox']
                vx2, vy2 = vx + vw, vy + vh

                intersection_x1 = max(x1, vx)
                intersection_y1 = max(y1, vy)
                intersection_x2 = min(x2, vx2)
                intersection_y2 = min(y2, vy2)

                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

                    if intersection_area > spot_area * 0.25:
                        spot['occupied'] = True
                        break

    def print_parking_status(self):
        if not self.parking_spots:
            return

        occupied_count = sum(1 for spot in self.parking_spots if spot['occupied'])
        total_spots = len(self.parking_spots)

        print("\n" + "=" * 50)
        print("PARKING STATUS")
        print("=" * 50)

        for spot in self.parking_spots:
            status = "ЗАНЯТО" if spot['occupied'] else "СВОБОДНО"
            print(f"Spot {spot['id']}: {status}")

        print("-" * 50)
        print(f"всего: {occupied_count}/{total_spots} занято")
        print(f"свободно: {total_spots - occupied_count} доступные места")
        print("=" * 50)

    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        return self.fps

    def draw_parking_info(self, frame):
        occupied_count = 0

        for spot in self.parking_spots:
            if 'points' in spot:
                # Рисуем полигональное парковочное место
                points = np.array(spot['points'], dtype=np.int32)

                if spot['occupied']:
                    color = (0, 0, 255)  # Красный для занятого
                    occupied_count += 1
                else:
                    color = (0, 255, 0)  # Зеленый для свободного

                # Рисуем заполненный полигон с прозрачностью
                overlay = frame.copy()
                cv2.fillPoly(overlay, [points], color)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Рисуем контур
                cv2.polylines(frame, [points], True, color, 3)

                # Добавляем номер места
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))

            else:
                # Для старых прямоугольных мест (обратная совместимость)
                x1, y1, x2, y2 = spot['coords']

                if spot['occupied']:
                    color = (0, 0, 255)
                    occupied_count += 1
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

            # Добавляем текст с номером места
            text = f"Spot {spot['id']}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            bg_x1 = center_x - text_size[0] // 2 - 5
            bg_y1 = center_y - text_size[1] // 2 - 5
            bg_x2 = center_x + text_size[0] // 2 + 5
            bg_y2 = center_y + text_size[1] // 2 + 5

            # Фон для текста
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(frame, text, (center_x - text_size[0] // 2, center_y + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        total_spots = len(self.parking_spots)
        free_spots = total_spots - occupied_count

        # Сохраняем и загружаем статус каждые 10 кадров
        if self.frame_count % 10 == 0 and total_spots > 0:
            self.save_current_status(occupied_count, total_spots)

        # Панель информации
        cv2.rectangle(frame, (5, 5), (500, 140), (0, 0, 0), -1)

        cv2.putText(frame, f"Parking: {free_spots}/{total_spots} free",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Статус FTP загрузки
        ftp_status = "ВКЛ" if self.upload_enabled else "ВЫКЛ"
        ftp_color = (0, 255, 0) if self.upload_enabled else (0, 0, 255)
        cv2.putText(frame, f"FTP: {ftp_status}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ftp_color, 2)

        # Время последней загрузки
        if self.last_upload_time > 0:
            last_upload_str = time.strftime('%H:%M:%S', time.localtime(self.last_upload_time))
            cv2.putText(frame, f"Last upload: {last_upload_str}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "LMB: add 4-point spot, 'c': clear, 's': save, 'u': toggle FTP, 'q': quit",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return occupied_count

    def run(self):
        if not self.download_yolo_files():
            return

        if not self.load_model():
            return

        self.load_parking_spots()

        # Проверяем подключение FTP
        if self.upload_enabled:
            print("\nПроверка FTP подключения...")
            test_data = [{"id": 1, "occupied": False}]
            if self.uploader.upload(test_data):
                print("✓ FTP подключение успешно!")
            else:
                print("✗ Проблемы с FTP подключением")
                choice = input("Продолжить без FTP? (y/n): ")
                if choice.lower() != 'y':
                    return
                self.upload_enabled = False

        # Создаем начальный файл статуса
        if self.parking_spots:
            self.save_current_status(0, len(self.parking_spots))
        else:
            self.save_current_status(0, 0)

        video_sources = [
            "test_images/parking.mp4",
            "parking.mp4",
            "video.mp4",
            0
        ]

        cap = None
        for source in video_sources:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"Successfully opened video source: {source}")
                break
            else:
                print(f"Failed to open: {source}")

        if not cap or not cap.isOpened():
            print("Failed to open any video source")
            return

        cv2.namedWindow('Parking Control System')
        cv2.setMouseCallback('Parking Control System', self.mouse_callback)

        print("\n" + "=" * 50)
        print("=== PARKING CONTROL SYSTEM ===")
        print("=" * 50)
        print("FTP Status:", "ENABLED" if self.upload_enabled else "DISABLED")
        if self.upload_enabled:
            print(f"FTP Host: {self.uploader.ftp_config['host']}")
            print(f"Upload interval: {self.upload_interval} seconds")
        print("\n=== CONTROLS ===")
        print("LMB - кликните 4 точки для парковочного места")
        print("'c' - очистить все парковочные места")
        print("'s' - сохранить парковочные места")
        print("'u' - включить/выключить FTP загрузку")
        print("'p' - показать статус парковки")
        print("'q' - выйти")
        print("==================\n")

        print("Статус парковки будет автоматически сохраняться в файл 'parking_status.json'")
        print("и загружаться на uCoz через FTP каждые 5 секунд")
        print("=" * 50 + "\n")

        frame_count = 0
        PROCESS_EVERY_N_FRAMES = 3

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break

                frame_count += 1
                self.frame_count = frame_count

                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    vehicles = self.detect_vehicles(frame, confidence_threshold=0.4)
                    self.check_parking_occupancy(vehicles)

                    if frame_count % 60 == 0:
                        self.print_parking_status()

                # Рисуем текущее парковочное место в процессе создания
                if self.drawing and self.current_spot:
                    # Рисуем уже добавленные точки
                    for i, point in enumerate(self.current_spot):
                        cv2.circle(frame, point, 5, (255, 255, 0), -1)
                        cv2.putText(frame, str(i + 1), (point[0] + 10, point[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Рисуем линии между точками
                    if len(self.current_spot) > 1:
                        for i in range(len(self.current_spot) - 1):
                            cv2.line(frame, self.current_spot[i], self.current_spot[i + 1], (255, 255, 0), 2)

                    # Рисуем линию от последней точки к курсору
                    if len(self.current_spot) < 4:
                        cv2.line(frame, self.current_spot[-1], self.current_mouse_pos, (255, 255, 0), 2)

                    # Показываем инструкцию
                    points_left = 4 - len(self.current_spot)
                    cv2.putText(frame, f"Click {points_left} more points",
                                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                occupied_count = self.draw_parking_info(frame)

                cv2.imshow('Parking Control System', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.parking_spots = []
                    self.save_parking_spots()
                    self.save_current_status(0, 0)
                    print("All parking spots cleared")
                elif key == ord('s'):
                    self.save_parking_spots()
                    if self.parking_spots:
                        occupied_count = sum(1 for spot in self.parking_spots if spot['occupied'])
                        self.save_current_status(occupied_count, len(self.parking_spots))
                elif key == ord('p'):
                    self.print_parking_status()
                elif key == ord('u'):
                    self.upload_enabled = not self.upload_enabled
                    status = "ВКЛЮЧЕНА" if self.upload_enabled else "ВЫКЛЮЧЕНА"
                    print(f"FTP загрузка {status}")
                    if self.upload_enabled:
                        # Принудительная загрузка при включении
                        if self.parking_spots:
                            occupied_count = sum(1 for spot in self.parking_spots if spot['occupied'])
                            self.save_current_status(occupied_count, len(self.parking_spots))

        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_parking_spots()

            # Сохраняем финальный статус
            if self.parking_spots:
                occupied_count = sum(1 for spot in self.parking_spots if spot['occupied'])
                self.save_current_status(occupied_count, len(self.parking_spots))

            print("\n" + "=" * 50)
            print("Программа завершена.")
            print(f"Локальный файл: '{self.status_file}'")
            if self.upload_enabled:
                print(f"Файл загружен на uCoz: {self.uploader.ftp_config['host']}")
            print("=" * 50)


def configure_ftp_settings():
    """Функция для настройки FTP параметров"""
    print("\n" + "=" * 50)
    print("НАСТРОЙКА FTP ДЛЯ uCoz")
    print("=" * 50)
    print("Для работы FTP загрузки выполните следующие шаги:")
    print("\n1. Войдите в панель управления uCoz")
    print("2. Перейдите в раздел 'Управление сайтом' → 'Файлы' → 'Доступ по FTP'")
    print("3. Создайте FTP аккаунт и запишите данные:")

    host = input("\nВведите FTP хост (например: вашсайт.ucoz.net): ").strip()
    username = input("Введите FTP логин: ").strip()
    password = input("Введите FTP пароль: ").strip()

    # Обновляем конфигурацию
    with open("ftp_config.json", "w") as f:
        json.dump({
            'host': host,
            'username': username,
            'password': password,
            'remote_path': '/parking_status.json'
        }, f, indent=2)

    print("\n✓ Настройки сохранены в ftp_config.json")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Проверяем наличие конфигурации FTP
    if not os.path.exists("ftp_config.json"):
        print("FTP конфигурация не найдена.")
        configure = input("Хотите настроить FTP сейчас? (y/n): ")
        if configure.lower() == 'y':
            configure_ftp_settings()
        else:
            print("FTP загрузка будет отключена. Вы можете настроить её позже.")

    # Запускаем детектор
    detector = ParkingDetector()

    # Загружаем настройки FTP если есть
    if os.path.exists("ftp_config.json"):
        try:
            with open("ftp_config.json", "r") as f:
                ftp_config = json.load(f)
            detector.uploader.ftp_config = ftp_config
            print("FTP настройки загружены из ftp_config.json")
        except:
            print("Ошибка загрузки FTP настроек")

    detector.run()