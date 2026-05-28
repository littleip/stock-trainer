"""
盘感训练器 - 通达信数据解析模块
"""

import struct
from datetime import datetime
from pathlib import Path


class TongdaxinData:
    """通达信本地数据读取器"""

    # 日线记录结构：日期(4) + 开(4) + 高(4) + 低(4) + 收(4) + 额(4) + 量(4) + 保留(4)
    RECORD_SIZE = 32
    RECORD_FORMAT = '<IIIIIfII'  # 小端序

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sh_dir = self.data_dir / 'sh' / 'lday'
        self.sz_dir = self.data_dir / 'sz' / 'lday'

    def get_stock_list(self) -> list:
        """获取A股股票列表（主板60/68、创业板30、中小板00）"""
        stocks = []
        if self.sh_dir.exists():
            for f in self.sh_dir.glob('sh*.day'):
                code = f.stem  # sh600000
                num = code[2:]
                # 沪市：60xxxx主板、68xxxx科创板
                if num.startswith('60') or num.startswith('68'):
                    stocks.append(code)
        if self.sz_dir.exists():
            for f in self.sz_dir.glob('sz*.day'):
                code = f.stem  # sz000001
                num = code[2:]
                # 深市：00xxxx主板、30xxxx创业板
                if num.startswith('00') or num.startswith('30'):
                    stocks.append(code)
        return sorted(stocks)

    def _detect_price_divisor(self, raw_records):
        """
        自动检测价格除数。
        通达信不同版本价格倍数不同：100、1000、10000
        策略：找一个收盘价在合理范围(1~50000)的记录来推断
        """
        for rec in raw_records[:100]:
            _, _, _, _, close_p, _, _, _ = rec
            if close_p <= 0:
                continue
            # 合理股价范围 1~50000
            if 1 <= close_p <= 50000:
                return 1       # 已经是真实价格
            elif 100 <= close_p <= 5000000:
                return 100
            elif 10000 <= close_p <= 500000000:
                return 1000
            elif 100000 <= close_p <= 5000000000:
                return 10000
        return 100  # 默认

    def read_day_file(self, stock_code: str) -> list:
        """读取单只股票的日线数据"""
        if stock_code.startswith('sh'):
            file_path = self.sh_dir / f'{stock_code}.day'
        elif stock_code.startswith('sz'):
            file_path = self.sz_dir / f'{stock_code}.day'
        else:
            raise ValueError(f"无效的股票代码格式: {stock_code}")

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 先读取所有原始记录
        raw_records = []
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(self.RECORD_SIZE)
                if len(data) < self.RECORD_SIZE:
                    break
                unpacked = struct.unpack(self.RECORD_FORMAT, data)
                raw_records.append(unpacked)

        # 自动检测价格除数
        divisor = self._detect_price_divisor(raw_records)

        records = []
        for unpacked in raw_records:
            date_int, open_p, high_p, low_p, close_p, amount, volume, _ = unpacked

            date_str = str(date_int)
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
            except:
                continue

            # 过滤无效数据
            if close_p <= 0 or volume <= 0:
                continue

            record = {
                'date': date,
                'open': open_p / divisor,
                'high': high_p / divisor,
                'low': low_p / divisor,
                'close': close_p / divisor,
                'amount': amount,
                'volume': volume
            }
            records.append(record)

        return records


def calculate_rsi(closes: list, period: int = 6) -> list:
    """
    计算RSI指标（Wilder平滑法）

    Args:
        closes: 收盘价列表
        period: RSI周期，默认6

    Returns:
        list: RSI值列表（前period个为None）
    """
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_values = [None] * period

    # 计算初始平均涨跌
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rsi_values.append(100 - 100 / (1 + avg_gain / avg_loss))

    # 后续用Wilder平滑
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rsi_values.append(100 - 100 / (1 + avg_gain / avg_loss))

    return rsi_values


if __name__ == '__main__':
    import yaml
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    td = TongdaxinData(config['tongdaxin_dir'])
    stocks = td.get_stock_list()
    print(f"找到 {len(stocks)} 只股票")
    if stocks:
        print(f"示例: {stocks[:5]}")
        # 读取第一只股票看看
        data = td.read_day_file(stocks[0])
        print(f"  {stocks[0]}: {len(data)} 条记录")
        if data:
            print(f"  最早: {data[0]['date']} 收盘: {data[0]['close']}")
            print(f"  最新: {data[-1]['date']} 收盘: {data[-1]['close']}")
