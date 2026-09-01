"""Точка входа приложения.

Порядок один и тот же при любом исходе: настройки → контроллер → окно → цикл
событий, а на выходе — `AppController.shutdown()` в `finally`. Именно поэтому
остановку делает `main`, а не окно: правило KB_05 №6 (`Stop` в `finally`)
обязано выполняться и когда окно не открылось вовсе.

Запуск::

    python main.py
"""

import sys

from fbg.io import config as config_module
from fbg.ui.app import AppController


def main(argv: list[str] | None = None) -> int:
    """Запускает приложение. Возвращает код выхода процесса."""
    # Qt импортируется здесь, а не наверху: `fbg.ui.app` без него обходится,
    # и это проверяется тестом. Ошибка настроек не должна требовать Qt.
    from PySide6.QtWidgets import QApplication

    from fbg.ui.main_window import MainWindow

    loaded = config_module.load()
    controller = AppController(loaded.config, config_path=loaded.path, issues=loaded.issues)

    application = QApplication(argv if argv is not None else sys.argv)
    try:
        controller.start()
        window = MainWindow(controller)
        window.start_updates()
        window.show()
        return application.exec()
    finally:
        for failure in controller.shutdown():
            print(f"при остановке: {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
