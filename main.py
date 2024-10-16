import sys
from PyQt5.QtWidgets import QApplication
from src.gui import MainWindow
from loguru import logger

logger.add("debug.log", level="DEBUG", rotation="3 MB", compression="zip")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
