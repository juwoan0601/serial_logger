import serial
import time
import csv
import os
from datetime import datetime
from threading import Thread, Event, Lock
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import serial.tools.list_ports
import webbrowser
from flask import Flask

# Version 2025.02.10

# ===== Serial Communication Settings =====
COM_PORT = 'COM82'        # Arduino serial port
BAUDRATE = 115200         # Communication speed (bps)
LOG_PERIOD = 10           # Logging period (ms)

# ===== Data Format Settings =====
NUM_VALUE = 3            # Number of values to receive per line
LABEL_VALUE = [          # Labels for each value
    "ACC_X",             # Channel 1 EMG signal
    "ACC_Y",             # Channel 2 EMG signal
    "ACC_Z",             # Channel 3 EMG signal
]
TYPE_VALUE = [           # Expected data type for each value
    int,                # EMG1: integer type
    int,                # EMG2: integer type
    int,                # EMG3: integer type
]

# ===== Visualization Settings =====
IS_PLOT = True          # Enable/disable real-time plotting
MAX_XPLOT_NUM = 5000    # Maximum number of points to show on plot
PLOT_UPDATE_MS = 100    # Plot update interval in milliseconds

def check_serial_port():
    """
    Check if the specified COM port is available and can be opened.
    
    Returns:
        tuple: (bool, str) - (Success status, Error message if any)
    """
    # List all available ports
    available_ports = [port.device for port in serial.tools.list_ports.comports()]
    
    if not available_ports:
        return False, "No serial ports found on the system."
    
    if COM_PORT not in available_ports:
        return False, f"Specified port {COM_PORT} not found. Available ports: {', '.join(available_ports)}"
    
    try:
        ser = serial.Serial(COM_PORT, BAUDRATE, timeout=0.1)
        ser.close()
        return True, f"Successfully connected to {COM_PORT}"
    except serial.SerialException as e:
        return False, f"Error accessing {COM_PORT}: {str(e)}"

class DataBuffer:
    """Thread-safe buffer for storing latest data values"""
    def __init__(self):
        self.lock = Lock()
        self.latest_values = [0] * NUM_VALUE
        self.has_new_data = False
        self.time_data = []
        self.value_data = [[] for _ in range(NUM_VALUE)]
        self.start_time = None
    
    def update(self, values):
        """Update buffer with new values"""
        with self.lock:
            current_time = time.time()
            if self.start_time is None:
                self.start_time = current_time
            
            elapsed_ms = int((current_time - self.start_time) * 1000)
            
            self.latest_values = values.copy()
            self.time_data.append(elapsed_ms)
            for i, value in enumerate(values):
                self.value_data[i].append(value)
            
            if len(self.time_data) > MAX_XPLOT_NUM:
                self.time_data.pop(0)
                for data_list in self.value_data:
                    data_list.pop(0)
            
            self.has_new_data = True
    
    def get(self):
        """Get latest values and reset new data flag"""
        with self.lock:
            self.has_new_data = False
            return {
                'time': self.time_data,
                'values': self.value_data,
                'latest': self.latest_values
            }

class SerialReader(Thread):
    """Thread for reading serial data"""
    def __init__(self, data_buffer, stop_event):
        super().__init__()
        self.data_buffer = data_buffer
        self.stop_event = stop_event
        self.ser = None
    
    def connect_serial(self):
        """Establish serial connection"""
        success, message = check_serial_port()
        if not success:
            raise serial.SerialException(message)
            
        self.ser = serial.Serial(COM_PORT, BAUDRATE, timeout=0.1)
        time.sleep(2)  # Wait for connection to stabilize
        self.ser.reset_input_buffer()
        print(message)
    
    def run(self):
        try:
            self.connect_serial()
            data_buffer = ""
            
            while not self.stop_event.is_set():
                if self.ser.in_waiting > 0:
                    byte_data = self.ser.read(self.ser.in_waiting)
                    data_buffer += byte_data.decode()
                    
                    while '\n' in data_buffer:
                        line, data_buffer = data_buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                            
                        try:
                            values = parse_serial_data(line)
                            self.data_buffer.update(values)
                        except ValueError as e:
                            print(f"Error parsing data: {e}")
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Serial reader error: {e}")
        finally:
            if self.ser:
                self.ser.close()

class DataLogger(Thread):
    """Thread for logging data at fixed intervals"""
    def __init__(self, data_buffer, stop_event, filepath):
        super().__init__()
        self.data_buffer = data_buffer
        self.stop_event = stop_event
        self.filepath = filepath
    
    def run(self):
        try:
            next_log_time = time.time()
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                if current_time < next_log_time:
                    time.sleep(0.001)
                    continue
                
                data = self.data_buffer.get()
                if data['time']:
                    elapsed_ms = data['time'][-1]
                    values = data['latest']
                    
                    with open(self.filepath, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([elapsed_ms] + values)
                    
                    value_pairs = [
                        f"{label}({type_val.__name__}): {value}" 
                        for label, type_val, value 
                        in zip(LABEL_VALUE, TYPE_VALUE, values)
                    ]
                    print(f"Time: {elapsed_ms}ms, {', '.join(value_pairs)}")
                
                next_log_time = current_time + (LOG_PERIOD / 1000.0)
                
        except Exception as e:
            print(f"Logger error: {e}")

def create_dash_app(data_buffer, stop_event):
    """Create and configure Dash application for real-time plotting"""
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Real-time Sensor Data"),
        html.Div([
            dcc.Graph(id='live-graph', style={'height': '800px'}),
            dcc.Interval(
                id='interval-component',
                interval=PLOT_UPDATE_MS,
                n_intervals=0
            )
        ])
    ])
    
    @app.callback(
        Output('live-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n):
        data = data_buffer.get()
        
        fig = make_subplots(rows=NUM_VALUE, cols=1,
                           subplot_titles=LABEL_VALUE,
                           shared_xaxes=True)
        
        for i in range(NUM_VALUE):
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['values'][i],
                    name=LABEL_VALUE[i],
                    mode='lines'
                ),
                row=i+1, col=1
            )
            
            # Add vertical line for current time
            if data['time']:
                fig.add_vline(
                    x=data['time'][-1],
                    line_color='black',
                    line_dash='dash',
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Sensor Data vs Time"
        )
        
        return fig
    
    return app

def main():
    """Main program execution"""
    try:
        # Validate settings
        validate_settings()
        
        # Create data folder and file
        data_dir = create_data_folder()
        filename = get_filename()
        filepath = os.path.join(data_dir, filename)
        
        # Write CSV header
        header = create_header()
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        
        print(f"Logging data to {filepath}")
        print(f"Headers: {', '.join(header)}")
        
        # Initialize shared objects
        stop_event = Event()
        data_buffer = DataBuffer()
        
        # Initialize and start threads
        serial_reader = SerialReader(data_buffer, stop_event)
        data_logger = DataLogger(data_buffer, stop_event, filepath)
        
        serial_reader.start()
        data_logger.start()
        
        # Initialize plotter if needed
        if IS_PLOT:
            app = create_dash_app(data_buffer, stop_event)
            webbrowser.open('http://127.0.0.1:8050/')  # Open browser automatically
            app.run_server(debug=False)
        
        # Wait for keyboard interrupt
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping all threads...")
        stop_event.set()
        
        if 'serial_reader' in locals():
            serial_reader.join()
        if 'data_logger' in locals():
            data_logger.join()
            
        print("All threads stopped")
        
    except Exception as e:
        print(f"Error in main: {e}")
        if 'stop_event' in locals():
            stop_event.set()
            
def validate_settings():
    """
    Validate all configuration settings including data types.
    
    Raises:
        ValueError: If configuration settings are invalid or inconsistent
    """
    if len(LABEL_VALUE) != NUM_VALUE:
        raise ValueError(
            f"LABEL_VALUE length({len(LABEL_VALUE)}) does not match NUM_VALUE({NUM_VALUE})"
        )
    
    if len(TYPE_VALUE) != NUM_VALUE:
        raise ValueError(
            f"TYPE_VALUE length({len(TYPE_VALUE)}) does not match NUM_VALUE({NUM_VALUE})"
        )
    
    for i, type_val in enumerate(TYPE_VALUE):
        if type_val not in [int, float, str]:
            raise ValueError(
                f"Invalid type {type_val} at index {i}. Must be int, float, or str"
            )

def create_data_folder():
    """
    Create 'data' folder for storing CSV files if it doesn't exist.
    
    Returns:
        str: Path to the data directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def get_filename():
    """
    Generate filename based on current timestamp.
    
    Returns:
        str: Filename in format "YYYYMMDD_HHMMSS.csv"
    """
    current_time = datetime.now()
    return current_time.strftime("%Y%m%d_%H%M%S.csv")

def create_header():
    """
    Create CSV header with time and value labels.
    
    Returns:
        list: Header row for CSV file
    """
    return ['time_ms'] + LABEL_VALUE

def convert_to_type(value, expected_type):
    """
    Convert string value to specified type.
    
    Args:
        value (str): Value to convert
        expected_type (type): Target type (int, float, or str)
    
    Returns:
        The converted value
        
    Raises:
        ValueError: If conversion fails
    """
    if expected_type == str:
        return value
    try:
        return expected_type(value)
    except ValueError:
        raise ValueError(f"Cannot convert '{value}' to {expected_type.__name__}")

def parse_serial_data(data_str):
    """
    Parse and validate serial data with type checking.
    
    Args:
        data_str (str): Raw data string from serial port
        
    Returns:
        list: Parsed and type-converted values
        
    Raises:
        ValueError: If parsing fails or type conversion fails
    """
    try:
        if not data_str:
            raise ValueError("Empty data string")
        
        # Split values and remove empty strings
        values = data_str.rstrip(',').split(',')
        values = [x.strip() for x in values if x.strip()]
        
        if len(values) != NUM_VALUE:
            raise ValueError(f"Expected {NUM_VALUE} values, got {len(values)}")
        
        # Convert each value to its specified type
        converted_values = []
        for value, expected_type in zip(values, TYPE_VALUE):
            converted_value = convert_to_type(value, expected_type)
            converted_values.append(converted_value)
        
        return converted_values
    
    except ValueError as e:
        raise ValueError(f"Invalid data format: {e}")

if __name__ == "__main__":
    main()