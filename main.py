"""
盘感训练器 v2 - PC端桌面应用
修复：历史K线显示、滑动窗口、均线、快捷键、统一记录
"""

import sys
import os
import json
import random
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QGroupBox,
    QLineEdit, QTextEdit, QFileDialog, QMessageBox, QShortcut,
    QDialog, QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QKeySequence, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from tdx_reader import TongdaxinData, calculate_rsi


# 窗口大小从 config.yaml 的 display_window 读取，运行时动态获取


class KLineChart(FigureCanvas):
    """K线图组件 - 通达信风格滑动窗口 + 鼠标拖动 + 十字线"""

    def __init__(self, parent=None, width=14, height=9):
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self._axes_list = []
        self._n = 0

    def plot_data(self, all_data, window_start, window_end, rsi_values,
                  buy_ops=None, sell_ops=None, stock_code='',
                  current_idx=None):
        """
        绘制固定窗口的K线图

        all_data: 全部K线数据 list of dict
        window_start, window_end: 显示窗口的起止索引（闭区间）
        rsi_values: 全部RSI值 list
        buy_ops/sell_ops: 买卖操作记录 list of dict (含 idx 字段)
        current_idx: 当前训练位置索引（提前结束时用于区分历史/未来K线）
        show_close_line: 是否在收盘价处画水平参考线
        """
        self.fig.clear()

        self._n = 0

        # 配色
        bg_color = '#1a1a2e'
        grid_color = '#2d2d44'
        text_color = '#e0e0e0'
        up_color = '#ef5350'     # 红涨
        down_color = '#26a69a'   # 绿跌
        ma5_color = '#ffeb3b'    # 黄色 MA5
        ma20_color = '#2196f3'   # 蓝色 MA20
        limit_up_color = '#e6a817'    # 涨停 土黄色
        limit_down_color = '#005f00'  # 跌停 深绿色
        future_alpha = 0.4  # 未来K线透明度

        # 三个子图比例 5:1:1
        gs = self.fig.add_gridspec(3, 1, height_ratios=[5, 1, 1], hspace=0.05,
                                    left=0.06, right=0.96, top=0.95, bottom=0.05)
        ax_kline = self.fig.add_subplot(gs[0])
        ax_vol = self.fig.add_subplot(gs[1], sharex=ax_kline)
        ax_rsi = self.fig.add_subplot(gs[2], sharex=ax_kline)
        self._axes_list = [ax_kline, ax_vol, ax_rsi]

        for ax in self._axes_list:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color, labelsize=8)
            ax.grid(True, color=grid_color, alpha=0.5, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color(grid_color)

        # 窗口内的数据
        w_data = all_data[window_start:window_end + 1]
        n = len(w_data)
        self._n = n
        x = np.arange(n)

        # ---- K线（蜡烛图）----
        for i in range(n):
            d = w_data[i]
            o, h, l, c = d['open'], d['high'], d['low'], d['close']
            global_idx = window_start + i
            color = up_color if c >= o else down_color
            alpha = 1.0

            # 涨跌停检测（对比前一日收盘价）
            if global_idx > 0:
                prev_close = all_data[global_idx - 1]['close']
                if prev_close > 0:
                    change_pct = (c - prev_close) / prev_close
                    # 涨停：涨幅>=9.9% 且不是一字板（open!=close 或 high!=low）
                    is_yizi = (o == h == l == c)
                    if change_pct >= 0.099 and not is_yizi:
                        color = limit_up_color
                    elif change_pct <= -0.099 and not is_yizi:
                        color = limit_down_color

            # 提前结束：未来K线半透明
            if current_idx is not None and global_idx > current_idx:
                alpha = future_alpha

            # 影线（细线）
            ax_kline.vlines(i, l, h, color=color, linewidth=0.8, alpha=alpha)

            # 实体（矩形）
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < (h - l) * 0.01:
                body_height = (h - l) * 0.01  # 十字星也画一点
            rect = matplotlib.patches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax_kline.add_patch(rect)

        # ---- MA均线 ----
        # 从窗口前多取一些数据来算MA（确保MA准确）
        ma_margin = 30
        ma_start = max(0, window_start - ma_margin)
        extended_closes = [all_data[j]['close'] for j in range(ma_start, window_end + 1)]
        offset = window_start - ma_start  # 窗口起点在extended中的偏移

        def calc_ma(data, period):
            """简单移动平均"""
            result = [None] * len(data)
            for i in range(period - 1, len(data)):
                result[i] = sum(data[i - period + 1:i + 1]) / period
            return result

        ma5_all = calc_ma(extended_closes, 5)
        ma20_all = calc_ma(extended_closes, 20)

        # 取窗口内的部分
        ma5_win = ma5_all[offset:offset + n]
        ma20_win = ma20_all[offset:offset + n]

        # 绘制MA5
        ma5_x = [i for i in range(n) if ma5_win[i] is not None]
        ma5_y = [ma5_win[i] for i in ma5_x]
        if ma5_y:
            ax_kline.plot(ma5_x, ma5_y, color=ma5_color, linewidth=1.0,
                         label='MA5', alpha=0.8)

        # 绘制MA20
        ma20_x = [i for i in range(n) if ma20_win[i] is not None]
        ma20_y = [ma20_win[i] for i in ma20_x]
        if ma20_y:
            ax_kline.plot(ma20_x, ma20_y, color=ma20_color, linewidth=1.0,
                         label='MA20', alpha=0.8)

        ax_kline.legend(loc='upper left', fontsize=8, facecolor=bg_color,
                       edgecolor=grid_color, labelcolor=text_color)

        # ---- 收盘价水平参考线（当前K线价位标注）----
        if current_idx is not None:
            close_x = current_idx - window_start
            if 0 <= close_x < n:
                close_price = w_data[close_x]['close']
                ax_kline.axhline(y=close_price, color='#ffffff', linewidth=0.6,
                                 linestyle='--', alpha=0.25)
                ax_kline.text(n - 1, close_price, f' {close_price:.2f}',
                             fontsize=7, color='#aaaaaa', va='center', ha='left')

        # ---- 标记买卖点 ----
        if buy_ops:
            for op in buy_ops:
                idx = op['idx'] - window_start
                if 0 <= idx < n:
                    price = w_data[idx]['low']
                    ax_kline.annotate('B', xy=(idx, price), xytext=(idx, price * 0.97),
                                     fontsize=10, color='#ff1744', fontweight='bold',
                                     ha='center', va='top',
                                     arrowprops=dict(arrowstyle='->', color='#ff1744', lw=1.5))

        if sell_ops:
            for op in sell_ops:
                idx = op['idx'] - window_start
                if 0 <= idx < n:
                    price = w_data[idx]['high']
                    ax_kline.annotate('S', xy=(idx, price), xytext=(idx, price * 1.03),
                                     fontsize=10, color='#00e676', fontweight='bold',
                                     ha='center', va='bottom',
                                     arrowprops=dict(arrowstyle='->', color='#00e676', lw=1.5))

        ax_kline.set_title(f'{stock_code}', fontsize=13, fontweight='bold',
                          color=text_color, loc='left')
        ax_kline.set_ylabel('价格', fontsize=9, color=text_color)
        ax_kline.set_xlim(-1, n)

        # ---- 成交量 ----
        for i in range(n):
            d = w_data[i]
            global_idx = window_start + i
            color = up_color if d['close'] >= d['open'] else down_color
            bar_alpha = future_alpha if (current_idx is not None and global_idx > current_idx) else 0.8
            ax_vol.bar(i, d['volume'], color=color, width=0.6, alpha=bar_alpha)

        ax_vol.set_ylabel('量', fontsize=9, color=text_color)
        # 只显示最高量的标签，避免拥挤
        max_vol = max(d['volume'] for d in w_data) if w_data else 1
        ax_vol.set_ylim(0, max_vol * 1.2)

        # ---- RSI ----
        rsi_win = rsi_values[window_start:window_end + 1]
        valid_rsi = [(i, r) for i, r in enumerate(rsi_win) if r is not None]
        if valid_rsi:
            rx, ry = zip(*valid_rsi)
            ax_rsi.plot(rx, ry, color='#ab47bc', linewidth=1.2)
            ax_rsi.axhline(y=30, color='#666666', linestyle='--', alpha=0.5, linewidth=0.8)
            ax_rsi.axhline(y=70, color='#666666', linestyle='--', alpha=0.5, linewidth=0.8)
            ax_rsi.fill_between(rx, 0, ry, where=[r < 30 for r in ry],
                               color='#26a69a', alpha=0.2)
            ax_rsi.fill_between(rx, 100, ry, where=[r > 70 for r in ry],
                               color='#ef5350', alpha=0.2)

        ax_rsi.set_ylabel('RSI(6)', fontsize=9, color=text_color)
        ax_rsi.set_ylim(0, 100)

        # ---- X轴日期标签 ----
        dates = [d['date'] for d in w_data]
        step = max(1, n // 12)
        tick_pos = list(range(0, n, step))
        tick_lab = [dates[i].strftime('%y/%m/%d') for i in tick_pos if i < n]
        ax_rsi.set_xticks(tick_pos[:len(tick_lab)])
        ax_rsi.set_xticklabels(tick_lab, rotation=45, fontsize=7, color=text_color)

        # 隐藏K线和成交量的x轴标签
        for label in ax_kline.get_xticklabels():
            label.set_visible(False)
        for label in ax_vol.get_xticklabels():
            label.set_visible(False)

        self.draw()

class ReasonDialog(QDialog):
    """
    理由选择弹窗 - 纯键盘操作
    ↑↓ 移动 | 空格 选中/取消 | Tab 切换到确定按钮 | 回车/空格 确定
    """

    def __init__(self, title, reasons, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(420, 480)
        self.selected_reasons = []

        font = QFont('宋体', 12)

        # 深色样式
        self.setStyleSheet("""
            QDialog { background-color: #1a1a2e; color: #e0e0e0; }
            QListWidget {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                font-family: '宋体', 'SimSun';
                font-size: 14px;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 14px;
                border-bottom: 1px solid #1a1a2e;
            }
            QListWidget::item:selected {
                background-color: #2d2d44;
                color: #ffffff;
                border: 2px solid #2196f3;
            }
            QPushButton {
                background-color: #2d2d44;
                color: #e0e0e0;
                border: 2px solid #3d3d5c;
                border-radius: 4px;
                padding: 12px;
                font-family: '宋体', 'SimSun';
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:focus {
                border: 3px solid #2196f3;
                background-color: #1565c0;
                color: white;
            }
            QLabel {
                color: #aaa;
                font-family: '宋体', 'SimSun';
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # 提示
        hint = QLabel('↑↓ 移动  空格 选中/取消  Tab 切换  回车 确定')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 列表（多选，用SingleSelection让当前行有高亮）
        self.list_widget = QListWidget()
        self.list_widget.setFont(font)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        for reason in reasons:
            item = QListWidgetItem(f'  ○  {reason}')
            item.setFont(QFont('宋体', 13))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(self.list_widget)

        # 确定按钮
        self.btn_ok = QPushButton('✓  确定')
        self.btn_ok.setFont(QFont('宋体', 14, QFont.Bold))
        self.btn_ok.setFocusPolicy(Qt.StrongFocus)
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._do_confirm)
        layout.addWidget(self.btn_ok)

        # 给列表安装事件过滤器，拦截键盘
        self.list_widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        """拦截列表的键盘事件"""
        if obj == self.list_widget and event.type() == event.KeyPress:
            key = event.key()
            current_row = self.list_widget.currentRow()

            if key == Qt.Key_Up:
                if current_row > 0:
                    self.list_widget.setCurrentRow(current_row - 1)
                return True

            if key == Qt.Key_Down:
                if current_row < self.list_widget.count() - 1:
                    self.list_widget.setCurrentRow(current_row + 1)
                else:
                    # 已到最后一个选项，自动跳到确定按钮
                    self.btn_ok.setFocus()
                return True

            if key == Qt.Key_Space:
                item = self.list_widget.item(current_row)
                name = item.text().split('  ', 2)[-1]
                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                    item.setText(f'  ○  {name}')
                else:
                    item.setCheckState(Qt.Checked)
                    item.setText(f'  ●  {name}')
                # 自动下移
                if current_row < self.list_widget.count() - 1:
                    self.list_widget.setCurrentRow(current_row + 1)
                return True

            if key == Qt.Key_Right:
                # → 快速跳到确定按钮
                self.btn_ok.setFocus()
                return True

            if key == Qt.Key_Tab:
                self.btn_ok.setFocus()
                return True

            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._do_confirm()
                return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        key = event.key()

        # 焦点在按钮上时
        if self.btn_ok.hasFocus():
            if key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
                self._do_confirm()
                return
            if key in (Qt.Key_Tab, Qt.Key_Left):
                self.list_widget.setFocus()
                return
            if key == Qt.Key_Escape:
                self.reject()
                return

        # 全局：Esc关闭
        if key == Qt.Key_Escape:
            self.reject()
            return

        super().keyPressEvent(event)

    def _do_confirm(self):
        """确认选择"""
        self.selected_reasons = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                text = item.text().split('  ', 2)[-1]
                self.selected_reasons.append(text)
        self.accept()

    def get_selected(self):
        return self.selected_reasons


class TrainingSession:
    """训练会话"""

    def __init__(self, stock_code, data, start_idx, train_days):
        self.stock_code = stock_code
        self.data = data
        self.start_idx = start_idx          # 第一根训练K线的索引
        self.train_days = train_days
        self.end_idx = min(start_idx + train_days - 1, len(data) - 1)

        self.current_idx = start_idx        # 当前显示到哪根K线
        self.operations = []
        self.position = None
        self.finished = False

    def get_display_range(self, display_window):
        """
        训练中：固定 display_window 根K线，末端=current_idx
        训练后：显示从起点到终点的全部K线（display_window + train_days）
        """
        if self.finished:
            # 从第一根展示的K线到训练最后一根
            win_start = max(0, self.start_idx - display_window)
            win_end = self.end_idx
        else:
            win_end = self.current_idx
            win_start = max(0, win_end - display_window + 1)
        return win_start, win_end

    def move_next(self):
        if self.current_idx < self.end_idx:
            self.current_idx += 1
            return True
        else:
            self.finished = True
            return False

    def move_prev(self):
        if self.current_idx > self.start_idx:
            self.current_idx -= 1
            return True
        return False

    def buy(self, reason, price=None):
        if self.position is not None:
            return False, "已经持仓，不能重复买入"
        current = self.data[self.current_idx]
        if price is None:
            price = current['close']
        self.position = {
            'type': 'long',
            'entry_price': price,
            'entry_idx': self.current_idx,
            'entry_date': current['date']
        }
        self.operations.append({
            'action': 'buy',
            'idx': self.current_idx,
            'date': current['date'].strftime('%Y-%m-%d'),
            'price': price,
            'reason': reason
        })
        return True, f"价格: {price:.2f}"

    def sell(self, reason, price=None):
        if self.position is None:
            return False, "当前空仓，不能卖出"
        current = self.data[self.current_idx]
        if price is None:
            price = current['close']
        profit_pct = (price - self.position['entry_price']) / self.position['entry_price'] * 100
        self.operations.append({
            'action': 'sell',
            'idx': self.current_idx,
            'date': current['date'].strftime('%Y-%m-%d'),
            'price': price,
            'reason': reason,
            'profit_pct': round(profit_pct, 2)
        })
        self.position = None
        return True, f"价格: {price:.2f} 盈亏: {profit_pct:+.2f}%"

    def get_result(self):
        """获取训练结果（不管是否提前结束都可用）"""
        # 如果还有持仓，按最后一天收盘价估算
        if self.position:
            final_price = self.data[self.current_idx]['close']
            profit_pct = (final_price - self.position['entry_price']) / self.position['entry_price'] * 100
        else:
            total_profit = 0
            for op in self.operations:
                if op['action'] == 'sell' and 'profit_pct' in op:
                    total_profit += op['profit_pct']
            profit_pct = total_profit

        # 计算每笔完整交易的盈亏
        trades = []
        for op in self.operations:
            if op['action'] == 'sell' and 'profit_pct' in op:
                trades.append(op['profit_pct'])

        win_count = len([t for t in trades if t > 0])
        loss_count = len([t for t in trades if t <= 0])
        max_drawdown = min(trades) if trades else 0

        return {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'stock_code': self.stock_code,
            'period_start': self.data[self.start_idx]['date'].strftime('%Y-%m-%d'),
            'period_end': self.data[self.current_idx]['date'].strftime('%Y-%m-%d'),
            'planned_end': self.data[self.end_idx]['date'].strftime('%Y-%m-%d'),
            'early_end': not self.finished,
            'total_operations': len(self.operations),
            'buy_count': len([o for o in self.operations if o['action'] == 'buy']),
            'sell_count': len([o for o in self.operations if o['action'] == 'sell']),
            'win_count': win_count,
            'loss_count': loss_count,
            'final_profit_pct': round(profit_pct, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_trade_pct': round(sum(trades) / len(trades), 2) if trades else 0,
            'operations': self.operations,
            'has_position': self.position is not None
        }


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.config = self.load_config()
        self.tdx_data = None
        self.stock_list = []
        self.current_session = None
        self.training_history = []

        self.init_ui()
        self.init_data()
        self.init_shortcuts()

    def load_config(self):
        config_path = Path(__file__).parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def init_shortcuts(self):
        """初始化快捷键"""
        # → : 下一根K线
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.on_next_day)
        # ← : 前一根K线
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev_day)
        # ↑ : 买入
        QShortcut(QKeySequence(Qt.Key_Up), self).activated.connect(self.on_buy)
        # ↓ : 卖出
        QShortcut(QKeySequence(Qt.Key_Down), self).activated.connect(self.on_sell)
        # Space : 观望
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.on_hold)
        # N : 新训练
        QShortcut(QKeySequence(Qt.Key_N), self).activated.connect(self.on_start_training)
        # Esc : 结束训练
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.on_end_training)
        # PageDown : 快进5天
        QShortcut(QKeySequence(Qt.Key_PageDown), self).activated.connect(self.on_fast_forward)

    def init_ui(self):
        self.setWindowTitle('盘感训练器 v2')
        self.setGeometry(50, 50, 1500, 950)

        # 深色主题
        self.setStyleSheet("""
            QMainWindow { background-color: #16213e; }
            QWidget { background-color: #16213e; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QGroupBox { 
                border: 1px solid #2d2d44; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 12px;
                color: #e0e0e0;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { 
                background-color: #2d2d44; 
                color: #e0e0e0; 
                border: 1px solid #3d3d5c;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #3d3d5c; }
            QPushButton:disabled { color: #666; background-color: #1a1a2e; }
            QComboBox { 
                background-color: #2d2d44; 
                color: #e0e0e0; 
                border: 1px solid #3d3d5c;
                padding: 4px;
            }
            QComboBox QAbstractItemView { background-color: #2d2d44; color: #e0e0e0; }
            QSpinBox { 
                background-color: #2d2d44; 
                color: #e0e0e0; 
                border: 1px solid #3d3d5c;
                padding: 4px;
            }
            QLineEdit { 
                background-color: #2d2d44; 
                color: #e0e0e0; 
                border: 1px solid #3d3d5c;
                padding: 4px;
            }
            QTextEdit { 
                background-color: #0f0f23; 
                color: #c0c0c0; 
                border: 1px solid #2d2d44;
                font-family: '宋体', 'SimSun', 'Courier New';
                font-size: 13px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ---- 左侧：图表区域 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部信息栏
        info_frame = QWidget()
        info_frame.setFixedHeight(36)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 2, 10, 2)

        self.label_stock = QLabel('股票: ---')
        self.label_stock.setFont(QFont('Microsoft YaHei', 13, QFont.Bold))
        self.label_date = QLabel('日期: ----/--/--')
        self.label_date.setFont(QFont('Microsoft YaHei', 11))
        self.label_price = QLabel('收盘: ----')
        self.label_price.setFont(QFont('Microsoft YaHei', 11))
        self.label_rsi = QLabel('RSI(6): --')
        self.label_rsi.setFont(QFont('Microsoft YaHei', 11))
        self.label_position = QLabel('持仓: 空仓')
        self.label_position.setFont(QFont('Microsoft YaHei', 11))
        self.label_progress = QLabel('进度: 0/0')
        self.label_progress.setFont(QFont('Microsoft YaHei', 11))

        for w in [self.label_stock, self.label_date, self.label_price,
                  self.label_rsi, self.label_position, self.label_progress]:
            info_layout.addWidget(w)
            info_layout.addWidget(QLabel('  '))
        info_layout.addStretch()

        left_layout.addWidget(info_frame)

        # K线图
        self.chart = KLineChart(self)
        left_layout.addWidget(self.chart)

        # 控制按钮栏
        control_frame = QWidget()
        control_frame.setFixedHeight(40)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 2, 10, 2)

        self.btn_prev = QPushButton('← 前一天 (Left)')
        self.btn_prev.clicked.connect(self.on_prev_day)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton('后一天 → (Right)')
        self.btn_next.clicked.connect(self.on_next_day)
        self.btn_next.setEnabled(False)

        self.btn_fast = QPushButton('快进 >> (PgDn)')
        self.btn_fast.clicked.connect(self.on_fast_forward)
        self.btn_fast.setEnabled(False)

        self.btn_end = QPushButton('结束训练 (Esc)')
        self.btn_end.clicked.connect(self.on_end_training)
        self.btn_end.setEnabled(False)

        self.btn_buy = QPushButton('↑ 买入')
        self.btn_buy.clicked.connect(self.on_buy)
        self.btn_buy.setEnabled(False)
        self.btn_buy.setStyleSheet(
            'QPushButton { background-color: #b71c1c; color: white; font-weight: bold; }'
            'QPushButton:disabled { background-color: #3d1a1a; color: #666; }')

        self.btn_sell = QPushButton('↓ 卖出')
        self.btn_sell.clicked.connect(self.on_sell)
        self.btn_sell.setEnabled(False)
        self.btn_sell.setStyleSheet(
            'QPushButton { background-color: #004d40; color: white; font-weight: bold; }'
            'QPushButton:disabled { background-color: #1a3d3a; color: #666; }')

        for w in [self.btn_prev, self.btn_next, self.btn_fast, self.btn_end,
                  QLabel('  |  '), self.btn_buy, self.btn_sell]:
            control_layout.addWidget(w)
        control_layout.addStretch()

        left_layout.addWidget(control_frame)

        # ---- 右侧：操作面板 ----
        right_widget = QWidget()
        right_widget.setFixedWidth(320)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 5, 5)

        # 快捷键说明
        hotkey_group = QGroupBox('快捷键')
        hotkey_layout = QVBoxLayout(hotkey_group)
        hotkey_text = (
            "→ 下一根K线\n"
            "← 前一根K线\n"
            "↑ 买入\n"
            "↓ 卖出\n"
            "空格 观望(下一天)\n"
            "PgDn 快进5天\n"
            "N 新训练\n"
            "Esc 结束训练"
        )
        lbl = QLabel(hotkey_text)
        lbl.setFont(QFont('Consolas', 9))
        lbl.setStyleSheet('color: #888;')
        hotkey_layout.addWidget(lbl)
        right_layout.addWidget(hotkey_group)

        # 训练配置
        config_group = QGroupBox('训练配置')
        config_layout = QVBoxLayout(config_group)

        config_layout.addWidget(QLabel('通达信数据目录:'))
        dir_layout = QHBoxLayout()
        self.edit_tdx_dir = QLineEdit(self.config.get('tongdaxin_dir', ''))
        btn_browse = QPushButton('浏览')
        btn_browse.clicked.connect(self.on_browse_dir)
        dir_layout.addWidget(self.edit_tdx_dir)
        dir_layout.addWidget(btn_browse)
        config_layout.addLayout(dir_layout)

        days_layout = QHBoxLayout()
        days_layout.addWidget(QLabel('训练天数:'))
        self.spin_days = QSpinBox()
        self.spin_days.setRange(10, 500)
        self.spin_days.setValue(self.config.get('train_days', 60))
        days_layout.addWidget(self.spin_days)
        config_layout.addLayout(days_layout)

        self.btn_start = QPushButton('开始新训练 (N)')
        self.btn_start.clicked.connect(self.on_start_training)
        self.btn_start.setStyleSheet(
            'QPushButton { background-color: #2e7d32; color: white; font-size: 14px; '
            'padding: 10px; font-weight: bold; }')
        config_layout.addWidget(self.btn_start)

        right_layout.addWidget(config_group)

        # 交易操作
        trade_group = QGroupBox('交易操作 (↑买入 / ↓卖出)')
        trade_layout = QVBoxLayout(trade_group)

        trade_hint = QLabel(
            '按 ↑ 弹出买入理由选择\n'
            '按 ↓ 弹出卖出理由选择\n'
            '可多选，空格勾选\n'
            'Tab切换到确定按钮'
        )
        trade_hint.setStyleSheet('color: #aaa; font-size: 13px; padding: 10px; font-family: "宋体", "SimSun";')
        trade_hint.setWordWrap(True)
        trade_layout.addWidget(trade_hint)

        right_layout.addWidget(trade_group)

        # 操作日志
        log_group = QGroupBox('操作日志')
        log_layout = QVBoxLayout(log_group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        right_layout.addWidget(log_group)

        # 组装
        main_layout.addWidget(left_widget, stretch=4)
        main_layout.addWidget(right_widget, stretch=1)

    def init_data(self):
        tdx_dir = self.edit_tdx_dir.text()
        if tdx_dir and Path(tdx_dir).exists():
            self.tdx_data = TongdaxinData(tdx_dir)
            self.stock_list = self.tdx_data.get_stock_list()
            self.log(f"加载完成，共 {len(self.stock_list)} 只股票")

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.text_log.append(f"[{timestamp}] {message}")

    def on_browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, '选择通达信vipdoc目录')
        if dir_path:
            self.edit_tdx_dir.setText(dir_path)
            self.init_data()

    def on_start_training(self):
        try:
            self._do_start_training()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            QMessageBox.critical(self, '错误', f'启动训练失败:\n{err}')
            self.log(f"错误: {e}")

    def _do_start_training(self):
        if not self.stock_list:
            QMessageBox.warning(self, '错误', '没有可用的股票数据，请检查通达信目录设置')
            return

        self.current_session = None  # 清理旧session

        stock_code = random.choice(self.stock_list)
        try:
            data = self.tdx_data.read_day_file(stock_code)
        except Exception as e:
            QMessageBox.warning(self, '错误', f'读取数据失败: {str(e)}')
            return

        train_days = self.spin_days.value()
        display_window = self.config.get('display_window', 120)
        # 需要至少 display_window 根历史数据
        min_start = display_window
        max_start = len(data) - train_days - 1

        if max_start <= min_start:
            self.log(f"股票 {stock_code} 数据不足({len(data)}条)，跳过")
            return

        start_idx = random.randint(min_start, max_start)
        self.current_session = TrainingSession(stock_code, data, start_idx, train_days)

        # 启用按钮
        for btn in [self.btn_next, self.btn_fast, self.btn_end,
                    self.btn_buy, self.btn_sell, self.btn_prev]:
            btn.setEnabled(True)

        self.log(f"{'='*40}")
        self.log(f"开始训练: {stock_code} ({len(data)}条数据)")
        s = self.current_session
        self.log(f"区间: {s.data[s.start_idx]['date'].strftime('%Y-%m-%d')} "
                 f"至 {s.data[s.end_idx]['date'].strftime('%Y-%m-%d')}")
        self.log(f"快捷键: →前进 ↑买入 ↓卖出 空格观望 Esc结束")
        self.log(f"{'='*40}")

        self.update_display()

    def update_display(self):
        if not self.current_session:
            return
        try:
            self._do_update_display()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            QMessageBox.critical(self, '绘图错误', f'刷新显示失败:\n{err}')

    def _do_update_display(self):

        s = self.current_session
        cur = s.data[s.current_idx]

        # 信息栏
        self.label_stock.setText(f"股票: {s.stock_code}")
        self.label_date.setText(f"日期: {cur['date'].strftime('%Y-%m-%d')}")
        self.label_price.setText(f"收盘: {cur['close']:.2f}")
        self.label_progress.setText(f"进度: {s.current_idx - s.start_idx + 1}/{s.end_idx - s.start_idx + 1}")

        # RSI（当前值，给信息栏用）
        closes_for_rsi = [d['close'] for d in s.data[:s.current_idx + 1]]
        rsi_all = calculate_rsi(closes_for_rsi, self.config.get('rsi_period', 6))
        current_rsi = rsi_all[-1] if rsi_all and rsi_all[-1] is not None else 0
        self.label_rsi.setText(f"RSI(6): {current_rsi:.1f}")

        # 持仓
        if s.position:
            pct = (cur['close'] - s.position['entry_price']) / s.position['entry_price'] * 100
            color = '#ef5350' if pct >= 0 else '#26a69a'
            self.label_position.setText(
                f"持仓: 成本{s.position['entry_price']:.2f} 盈亏{pct:+.2f}%")
            self.label_position.setStyleSheet(f"QLabel {{ color: {color}; }}")
        else:
            self.label_position.setText("持仓: 空仓")
            self.label_position.setStyleSheet("QLabel { color: #888; }")

        # 固定窗口
        dw = self.config.get('display_window', 120)
        win_start, win_end = s.get_display_range(dw)
        all_closes = [d['close'] for d in s.data[:win_end + 1]]
        rsi_full = calculate_rsi(all_closes, self.config.get('rsi_period', 6))
        buy_ops = [o for o in s.operations if o['action'] == 'buy']
        sell_ops = [o for o in s.operations if o['action'] == 'sell']

        self.chart.plot_data(
            s.data, win_start, win_end, rsi_full,
            buy_ops, sell_ops, s.stock_code,
            current_idx=s.current_idx
        )

    def on_prev_day(self):
        if self.current_session and self.current_session.move_prev():
            self.update_display()

    def on_next_day(self):
        if not self.current_session:
            return
        if self.current_session.move_next():
            self.update_display()
        else:
            self.log("已到训练结束日期")
            self.on_end_training()

    def on_fast_forward(self):
        if not self.current_session:
            return
        for _ in range(5):
            if not self.current_session.move_next():
                self.on_end_training()
                return
        self.update_display()

    def on_buy(self):
        if not self.current_session:
            return
        s = self.current_session
        if s.position is not None:
            self.log("⚠ 已持仓，不能重复买入")
            return
        # T+1：卖出当天不能再买入
        if s.operations and s.operations[-1]['action'] == 'sell':
            last_sell_date = s.operations[-1]['date']
            today = s.data[s.current_idx]['date'].strftime('%Y-%m-%d')
            if last_sell_date == today:
                self.log("⚠ T+1规则：卖出当天不能买入")
                return
        reasons = self.config.get('buy_reasons', ['RSI超卖', '放量上涨', '直觉盘感'])
        dlg = ReasonDialog('选择买入理由', reasons, self)
        if dlg.exec_() == QDialog.Accepted:
            selected = dlg.get_selected()
            if not selected:
                selected = ['未选择理由']
            reason = ' | '.join(selected)
            success, msg = self.current_session.buy(reason)
            if success:
                self.log(f"▲ 买入 {msg} | 理由: {reason}")
            else:
                self.log(f"买入失败: {msg}")
            self.update_display()

    def on_sell(self):
        if not self.current_session:
            return
        s = self.current_session
        if s.position is None:
            self.log("⚠ 当前空仓，不能卖出")
            return
        # T+1：买入当天不能卖出
        today = s.data[s.current_idx]['date'].strftime('%Y-%m-%d')
        if s.position['entry_date'].strftime('%Y-%m-%d') == today:
            self.log("⚠ T+1规则：买入当天不能卖出")
            return
        reasons = self.config.get('sell_reasons', ['RSI超买', '触及压力位', '直觉盘感'])
        dlg = ReasonDialog('选择卖出理由', reasons, self)
        if dlg.exec_() == QDialog.Accepted:
            selected = dlg.get_selected()
            if not selected:
                selected = ['未选择理由']
            reason = ' | '.join(selected)
            success, msg = self.current_session.sell(reason)
            if success:
                self.log(f"▼ 卖出 {msg} | 理由: {reason}")
            else:
                self.log(f"卖出失败: {msg}")
            self.update_display()

    def on_hold(self):
        if self.current_session:
            self.log("— 观望")
            self.on_next_day()

    def on_end_training(self):
        if not self.current_session:
            return

        # 如果还有持仓，按当前收盘价自动卖出
        s = self.current_session
        if s.position is not None:
            cur = s.data[s.current_idx]
            price = cur['close']
            profit_pct = (price - s.position['entry_price']) / s.position['entry_price'] * 100
            s.operations.append({
                'action': 'sell',
                'idx': s.current_idx,
                'date': cur['date'].strftime('%Y-%m-%d'),
                'price': price,
                'reason': '训练结束自动平仓',
                'profit_pct': round(profit_pct, 2)
            })
            s.position = None
            self.log(f"▼ 训练结束自动平仓 | 价格: {price:.2f} 盈亏: {profit_pct:+.2f}%")

        result = s.get_result()
        self.training_history.append(result)

        # 保存到统一JSON文件
        self.save_all_records()

        # 显示结果
        self.log("")
        self.log("=" * 45)
        self.log(f"训练结束: {result['stock_code']}")
        self.log(f"区间: {result['period_start']} 至 {result['period_end']}")
        if result['early_end']:
            self.log(f"（提前结束，原计划至 {result['planned_end']}）")
        self.log(f"买入 {result['buy_count']} 次 | 卖出 {result['sell_count']} 次")
        self.log(f"胜 {result['win_count']} | 负 {result['loss_count']}")
        self.log(f"最终盈亏: {result['final_profit_pct']:+.2f}%")
        self.log(f"最大单笔回撤: {result['max_drawdown']:.2f}%")
        if result['has_position']:
            self.log("⚠ 训练结束时仍有持仓（按收盘价估算）")
        self.log("=" * 45)
        self.log("")

        # 禁用按钮
        for btn in [self.btn_next, self.btn_fast, self.btn_end,
                    self.btn_buy, self.btn_sell, self.btn_prev]:
            btn.setEnabled(False)

        # 快速逐条显示剩余K线直到最后一天
        while s.current_idx < s.end_idx:
            s.current_idx += 1
        s.finished = True

        # 刷新显示（展示全部K线）
        self.update_display()

    def save_all_records(self):
        """所有训练记录保存到一个JSON文件"""
        records_dir = Path(__file__).parent / 'records'
        records_dir.mkdir(exist_ok=True)

        filepath = records_dir / 'all_training_records.json'

        # 读取已有记录
        existing = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                existing = []

        # 添加新记录
        existing.append(self.training_history[-1])

        # 写回
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        self.log(f"记录已保存至 records/all_training_records.json (共{len(existing)}条)")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
