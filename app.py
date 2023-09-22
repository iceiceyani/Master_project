import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

class MyApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create and configure the main window
        self.setWindowTitle("My Application")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout()

        # Create a "Browse" button
        browse_button = QPushButton("Browse Excel File")
        browse_button.clicked.connect(self.browse_file)

        # Create a "Run Analysis" button
        run_button = QPushButton("Run Analysis")
        run_button.clicked.connect(self.run_analysis)

        # Add the buttons to the layout
        layout.addWidget(browse_button)
        layout.addWidget(run_button)

        # Set the layout for the central widget
        central_widget.setLayout(layout)

        # Initialize a variable to store the selected file path
        self.file_path = ""

    @pyqtSlot()
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Excel Files (*.xlsx *.xls)")
        selected_file, _ = file_dialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")

        if selected_file:
            self.file_path = selected_file
            print(f"Selected file: {self.file_path}")

    @pyqtSlot()
    def run_analysis(self):
        if self.file_path:
            try:
                # Set the current working directory to your project directory
                os.chdir("C:/Master_project")

                # Run the analysis script with the selected Excel file path
                subprocess.call([sys.executable, "main.py", self.file_path])
            except Exception as e:
                print(f"Error running analysis: {str(e)}")
        else:
            print("Please select an Excel file before running the analysis.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApplication()
    window.show()
    sys.exit(app.exec_())
