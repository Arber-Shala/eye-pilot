import argparse
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import logging

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()



def main():
    print()
    print(int(BoardIds.MUSE_2_BOARD))
    '''
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    board_shim = BoardShim(BoardIds.MUSE_2_BOARD, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000) # ,args.steamer_params
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

    
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()
    board.start_stream(45000)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(5)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    feature_vector = bands[0]
    print(feature_vector)

    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
    mindfulness.release()

    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
    restfulness.release()
    '''

if __name__ == "__main__":
    main()