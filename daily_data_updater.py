from datetime import datetime
import os
import dotenv
dotenv.load_dotenv()

import pandas as pd
from datetime import datetime
from typing import Any

import dotenv
dotenv.load_dotenv()

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_stock_daily_data_to_postgres(date: str | datetime | None, df, connection_string: str, target_table_name: str, percent_ratio: int = 100):
    """
    将股票日线数据上传到 PostgreSQL 数据库。
    """
    
    def convert_stock_daily_df_to_sql_df(date: str | datetime | None, df, percent_ratio: int = 100):
        """
        将股票日线数据转换为 SQL 数据。
        """
        sql_df = df.copy()
        sql_df = sql_df.reset_index(drop=False)
        
        # 处理 date 参数：如果为 None，尝试从 sql_df 中获取
        if date is not None:
            sql_df['date'] = date
        elif 'date' not in sql_df.columns:
            raise ValueError("date parameter is None and 'date' column not found in df")
        
        # 格式化日期并生成 date_str
        sql_df['date'] = pd.to_datetime(sql_df['date'])
        date_str = sql_df['date'].dt.strftime('%Y%m%d')
        
        sql_df["id"] = (
            date_str + "_" + 
            sql_df["symbol"].astype(str)
        )
        print(f"sql_df: \n{sql_df.columns}")
        sql_df["symbol"] = sql_df["symbol"].astype(str)
        sql_df["name"] = ""
        sql_df["exchange"] = ""
        if "close" in sql_df.columns:
            sql_df["close"] = pd.to_numeric(sql_df["close"], errors="coerce")
        else:
            sql_df["close"] = pd.to_numeric(sql_df["price"], errors="coerce")
            
        sql_df["change"] = pd.to_numeric(sql_df["change"], errors="coerce")
        if "change_rate" in sql_df.columns:
            sql_df["change_percent"] = pd.to_numeric(sql_df["change_rate"], errors="coerce")
        elif "change_percent" in sql_df.columns:
            sql_df["change_percent"] = pd.to_numeric(sql_df["change_percent"], errors="coerce") / percent_ratio

        sql_df["volume"] = pd.to_numeric(sql_df["volume"], errors="coerce")
        sql_df["amount"] = pd.to_numeric(sql_df["amount"], errors="coerce")
        sql_df["high"] = pd.to_numeric(sql_df["high"], errors="coerce")
        sql_df["low"] = pd.to_numeric(sql_df["low"], errors="coerce")
        sql_df["open"] = pd.to_numeric(sql_df["open"], errors="coerce")
        sql_df["pre_close"] = pd.to_numeric(sql_df["pre_close"], errors="coerce")
        sql_df["amplitude"] = pd.to_numeric(sql_df["amplitude"], errors="coerce")
        if "volume_ratio" in sql_df.columns:
            sql_df["volume_ratio"] = pd.to_numeric(sql_df["volume_ratio"], errors="coerce")
        else:
            sql_df["volume_ratio"] = None
        if "turnover_rate" in sql_df.columns:
            sql_df["turnover_rate"] = pd.to_numeric(sql_df["turnover_rate"], errors="coerce")
        else:
            sql_df["turnover_rate"] = None
        sql_df["created_at"] = datetime.now()
        sql_df["updated_at"] = datetime.now()
        sql_df['date'] = pd.to_datetime(sql_df['date'])
        return sql_df[['id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'amount', 'volume', 'change', 'change_percent', 'pre_close', 'amplitude', 'turnover_rate', 'created_at', 'updated_at']]

    sql_df = convert_stock_daily_df_to_sql_df(date, df, percent_ratio)
    print(f"sql_df: \n{sql_df}")

    from sqlalchemy import create_engine, text
    from sqlalchemy.dialects.postgresql import insert

    engine = create_engine(connection_string)
    def upsert_on_conflict(table, conn, keys, data_iter):
        """
        自定义 pandas 写入方法：实现 PostgreSQL 的 Upsert
        """
        # 1. 构建基础插入语句
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        
        # 2. 定义冲突处理逻辑 (基于 'id' 列)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['id'], # 这里必须是数据库里的主键或唯一约束列名
            set_={
                'open': stmt.excluded.open,
                'high': stmt.excluded.high,
                'low': stmt.excluded.low,
                'close': stmt.excluded.close,
                'amount': stmt.excluded.amount,
                'volume': stmt.excluded.volume,
                'change': stmt.excluded.change,
                'change_percent': stmt.excluded.change_percent,
                'pre_close': stmt.excluded.pre_close,
                'updated_at': stmt.excluded.updated_at
            }
        )
        conn.execute(upsert_stmt)
    
    sql_df = convert_stock_daily_df_to_sql_df(date, df, percent_ratio)
    sql_df = sql_df.dropna(subset=['open', 'close', 'volume', 'amount', 'high', 'low', 'change'])
    sql_df.to_sql(
        target_table_name, 
        engine, 
        if_exists='append', # 必须设为 append 才能触发自定义 upsert 逻辑
        index=False, 
        method=upsert_on_conflict, # 引用上面的函数
        chunksize=1000
    )
    engine.dispose()
    
def upload_prediction_data_to_postgres(model_name: str, pred_df: pd.DataFrame, connection_string: str, target_table_name: str):
    """
    将股票预测数据上传到 PostgreSQL 数据库。
    """
    import pandas as pd
    from sqlalchemy import create_engine, text
    from sqlalchemy.dialects.postgresql import insert

    engine = create_engine(connection_string)
    def convert_prediction_df_to_sql_df(model_name, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        将股票预测数据转换为 SQL 数据。
        """
        sql_pred_df = pred_df.copy()
        sql_pred_df = sql_pred_df.reset_index(drop=False)
        sql_pred_df["id"] = (
            f"{model_name}_" + 
            sql_pred_df["datetime"].dt.strftime('%Y%m%d') + "_" +
            sql_pred_df["instrument"].astype(str)
        )
        sql_pred_df["model_name"] = model_name
        sql_pred_df["trading_date"] = sql_pred_df["datetime"]
        sql_pred_df = sql_pred_df.rename(columns={
            "datetime": "prediction_date",
            "score": "prediction_score",
            "instrument": "symbol"
        })
        sql_pred_df["name"] = ""
        sql_pred_df["exchange"] = ""

        sql_pred_df["trading_date"] = sql_pred_df["prediction_date"]
        sql_pred_df["created_at"] = datetime.now()
        sql_pred_df["updated_at"] = datetime.now()
        sql_pred_df['prediction_date'] = pd.to_datetime(sql_pred_df['prediction_date'])
        sql_pred_df['rank'] = sql_pred_df.groupby('prediction_date')['prediction_score'].rank(
            ascending=False,
            method='min'
        ).astype(int)
        if 'index' in sql_pred_df.columns:
            sql_pred_df = sql_pred_df.drop(columns=['index'])
        return sql_pred_df

    def upsert_on_conflict(table, conn, keys, data_iter):
        """
        自定义 pandas 写入方法：实现 PostgreSQL 的 Upsert
        """
        # 1. 构建基础插入语句
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        
        # 2. 定义冲突处理逻辑 (基于 'id' 列)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['id'], # 这里必须是数据库里的主键或唯一约束列名
            set_={
                'prediction_score': stmt.excluded.prediction_score,
                'rank': stmt.excluded.rank,
                'updated_at': stmt.excluded.updated_at
            }
        )
        conn.execute(upsert_stmt)

    # --- 写入数据库 ---
    # chunksize 设为 5000 左右可以平衡速度和内存
    sql_pred_df = convert_prediction_df_to_sql_df(model_name, pred_df)
    sql_pred_df.to_sql(
        target_table_name, 
        engine, 
        if_exists='append', # 必须设为 append 才能触发自定义 upsert 逻辑
        index=False, 
        method=upsert_on_conflict, # 引用上面的函数
        chunksize=5000
    )
    engine.dispose()


def gtimg_get_stock_current_prices(stock_codes: list[str]) -> dict[str, dict[str, float | int | str]]:
    """
    获取指定多个股票代码的当前价格及常用股票信息。
    
    Args:
        stock_codes: 股票代码列表，支持格式如 ["600000", "000001"] 或 ["SH600000", "SZ000001"]
    
    Returns:
        嵌套字典，第一层键为股票代码（格式化为 SH/SZ 前缀），第二层为包含以下字段的字典：
        - open: 今开价
        - close: 当前价格（最新价）
        - high: 最高价
        - low: 最低价
        - change: 涨跌额
        - change_percent: 涨跌幅（百分比）
        - amount: 成交额
        - volume: 成交量（手）
        - pre_close: 昨收价
        - name: 股票名称
    
    Example:
        >>> prices = gtimg_get_stock_current_prices(["600000", "000001"])
        >>> print(prices)
        {
            'SH600000': {
                'open': 11.54,
                'close': 11.54,
                'high': 11.57,
                'low': 11.46,
                'change': 0.00,
                'change_percent': 0.00,
                'amount': 487401034.0,
                'volume': 423616,
                'pre_close': 11.54,
                'name': '浦发银行'
            },
            'SZ000001': {...}
        }
    """
    if not stock_codes:
        return {}
    
    def _convert_to_api_format(code: str) -> str:
        """将股票代码转换为API需要的格式（sh600000, sz000001）"""
        code = str(code).strip().upper()
        # 如果已经有前缀，去掉前缀
        if code.startswith("SH"):
            return "sh" + code[2:]
        elif code.startswith("SZ"):
            return "sz" + code[2:]
        elif code.startswith("BJ"):
            return "bj" + code[2:]
        else:
            # 根据代码首字符判断
            if code[0] in ("0", "3"):  # 深市
                return "sz" + code
            elif code[0] == "6":  # 沪市
                return "sh" + code
            elif code[0] in ("4", "8", "9"):  # 北交所
                return "bj" + code
            else:
                # 默认当作沪市处理
                return "sh" + code
    
    def _safe_float(value: str, default: float = 0.0) -> float:
        """安全地将字符串转换为浮点数"""
        try:
            return float(value) if value and value.strip() else default
        except (ValueError, TypeError):
            return default
    
    def _safe_int(value: str, default: int = 0) -> int:
        """安全地将字符串转换为整数"""
        try:
            return int(float(value)) if value and value.strip() else default
        except (ValueError, TypeError):
            return default
    
    # 转换为API格式
    api_codes = [_convert_to_api_format(code) for code in stock_codes]
    # 构建URL
    url = f"https://qt.gtimg.cn/q={','.join(api_codes)}"
    
    try:
        with httpx.Client(headers=EM_HTTP_HEADERS_DEFAULT, timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.text
    except Exception as e:
        logger.error(f"Failed to fetch stock prices from {url}: {e}")
        return {}
    
    # 解析返回数据
    # 格式：v_sh600000="1~浦发银行~600000~11.54~11.54~11.56~..."; v_sz000001="51~平安银行~000001~11.46~...";
    # 字段索引说明（根据腾讯股票接口）：
    # 0: 未知
    # 1: 股票名称
    # 2: 股票代码
    # 3: 当前价格（最新价/收盘价）
    # 4: 昨收价
    # 5: 今开价
    # 6: 成交量（手）
    # 31: 最高价
    # 32: 最低价
    # 33: 涨跌额
    # 34: 涨跌幅（百分比，需要除以100）
    # 37: 成交额（元）
    result = {}
    
    # 按分号分割每条股票数据
    lines = content.split(';')
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('v_'):
            continue
        
        try:
            # 提取变量名和数据部分
            # v_sh600000="1~浦发银行~600000~11.54~..."
            parts = line.split('=', 1)
            if len(parts) != 2:
                continue
            
            var_name = parts[0].strip()  # v_sh600000
            data_str = parts[1].strip().strip('"')  # 1~浦发银行~600000~11.54~...
            
            # 从变量名提取股票代码（去掉 v_ 前缀）
            api_code = var_name[2:]  # sh600000
            
            # 解析数据，用 ~ 分隔
            data_fields = data_str.split('~')
            if len(data_fields) < 35:
                logger.warning(f"Insufficient fields for {api_code}, got {len(data_fields)} fields")
                continue
            
            # 将API格式的代码转换为标准格式
            # sh600000 -> SH600000, sz000001 -> SZ000001
            if api_code.startswith('sh'):
                standard_code = 'SH' + api_code[2:]
            elif api_code.startswith('sz'):
                standard_code = 'SZ' + api_code[2:]
            elif api_code.startswith('bj'):
                standard_code = 'BJ' + api_code[2:]
            else:
                standard_code = api_code.upper()
            
            # 提取各个字段
            # 根据腾讯股票接口返回格式：
            # 0: 未知
            # 1: 股票名称
            # 2: 股票代码
            # 3: 当前价格（最新价）
            # 4: 昨收价
            # 5: 今开价
            # 6: 成交量（手）
            # 31: 涨跌额
            # 32: 涨跌幅（百分比）
            # 33: 最高价
            # 34: 最低价
            # 35: 当前价/成交量/成交额（格式：价格/成交量/成交额，需要解析）
            # 37: 成交额（元，也可能在索引35中）
            
            name = data_fields[1] if len(data_fields) > 1 else ""  # 股票名称
            current_price = _safe_float(data_fields[3])  # 当前价格（最新价）
            pre_close = _safe_float(data_fields[4]) if len(data_fields) > 4 else current_price  # 昨收价
            open_price = _safe_float(data_fields[5]) if len(data_fields) > 5 else current_price  # 今开价
            volume = _safe_int(data_fields[6]) if len(data_fields) > 6 else 0  # 成交量（手）
            
            # 涨跌额和涨跌幅
            change = _safe_float(data_fields[31]) if len(data_fields) > 31 else 0.0  # 涨跌额
            change_percent = _safe_float(data_fields[32]) if len(data_fields) > 32 else 0.0  # 涨跌幅（百分比）
            
            # 最高价和最低价
            high_price = _safe_float(data_fields[33]) if len(data_fields) > 33 else current_price  # 最高价
            low_price = _safe_float(data_fields[34]) if len(data_fields) > 34 else current_price  # 最低价
            
            # 成交额：可能在索引35（格式：价格/成交量/成交额）或索引37
            amount = 0.0
            if len(data_fields) > 35 and data_fields[35]:
                # 尝试从索引35解析（格式：价格/成交量/成交额）
                amount_str = data_fields[35]
                if '/' in amount_str:
                    parts = amount_str.split('/')
                    if len(parts) >= 3:
                        amount = _safe_float(parts[2])
            if amount == 0.0 and len(data_fields) > 37:
                # 如果索引35解析失败，尝试索引37
                amount = _safe_float(data_fields[37])
            
            # 如果涨跌幅为空或0，尝试计算
            if change_percent == 0.0 and pre_close > 0 and current_price != pre_close:
                change_percent = ((current_price - pre_close) / pre_close) * 100
            
            # 如果涨跌额为空或0，尝试计算
            if change == 0.0 and current_price != pre_close:
                change = current_price - pre_close
            
            result[standard_code] = {
                "name": name,
                "open": open_price,
                "price": current_price,
                "close": current_price,  # 当前价格作为收盘价
                "high": high_price,
                "low": low_price,
                "change": change,
                "change_percent": change_percent,
                "amount": amount,
                "volume": volume,
                "pre_close": pre_close,
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse line: {line[:50]}... Error: {e}")
            continue
    
    return result
    
def _is_stock_code(s: str) -> bool:
    """判断是否为有效股票代码（如 SH600000、SZ000001、BJ430047），跳过分隔符如 --------。"""
    if not s or len(s) < 5:
        return False
    s = s.strip().upper()
    if s.startswith(("SH", "SZ", "BJ")):
        return s[2:].isdigit()
    return False


def _load_symbols_from_file(symbols_path: str) -> list[str]:
    """从 stock_symbols.txt 加载股票代码列表，每行一个（如 SZ300846、SH600000），跳过非代码行如 --------。"""
    path = os.path.abspath(symbols_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"股票清单文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and _is_stock_code(ln.strip())]
    return lines


def _fetch_current_prices_by_gtimg(symbols_path: str, batch_size: int = 100) -> tuple[pd.DataFrame, int]:
    """
    按 stock_symbols.txt 清单，每批 batch_size 只调用 gtimg_get_stock_current_prices，
    合并结果为与 em_retrieve_stock_rank_current 兼容的 DataFrame。
    返回 (final_data, response_total)。
    """
    symbols = _load_symbols_from_file(symbols_path)
    if not symbols:
        return pd.DataFrame(), 0

    all_prices: dict[str, dict] = {}
    max_attempts = 3
    retry_delay = 2.0
    print(f"{datetime.now()} start fetch current prices by gtimg, symbols count: {len(symbols)}")
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, len(symbols))
        for attempt in range(1, max_attempts + 1):
            try:
                batch_prices = gtimg_get_stock_current_prices(batch)
                if batch_prices:
                    all_prices.update(batch_prices)
                    break
                logger.warning(
                    "gtimg batch %d-%d attempt %d/%d returned empty",
                    batch_start, batch_end, attempt, max_attempts,
                )
            except Exception as e:
                logger.warning(
                    "gtimg batch %d-%d attempt %d/%d failed: %s",
                    batch_start, batch_end, attempt, max_attempts, e,
                )
            if attempt < max_attempts:
                time.sleep(retry_delay)
        else:
            logger.error("gtimg batch %d-%d failed after %d attempts", batch_start, batch_end, max_attempts)
        time.sleep(1)
    print(f"{datetime.now()} end fetch current prices by gtimg, symbols count: {len(symbols)}")

    if not all_prices:
        return pd.DataFrame(), 0

    rows = []
    for symbol, info in all_prices.items():
        # gtimg 返回 close/price, open, high, low, volume, amount, pre_close, change, change_percent
        change_rate = info.get("change_percent", 0.0)
        if isinstance(change_rate, (int, float)) and abs(change_rate) > 1:
            change_rate = change_rate / 100.0  # 若为百分比数值则转为小数
        pre_close = info.get("pre_close") or 0.0
        high = info.get("high", 0.0)
        low = info.get("low", 0.0)
        amplitude = (high - low) / pre_close if pre_close and pre_close > 0 else 0.0
        rows.append({
            "symbol": symbol,
            "price": info.get("close") or info.get("price", 0.0),
            "open": info.get("open", 0.0),
            "high": high,
            "low": low,
            "volume": info.get("volume", 0),
            "amount": info.get("amount", 0.0),
            "pre_close": pre_close,
            "change": info.get("change", 0.0),
            "change_rate": change_rate,
            "amplitude": amplitude,
            "name": info.get("name", ""),
        })
    final_data = pd.DataFrame(rows)
    response_total = len(final_data)
    return final_data, response_total

    
today_date = datetime.now().date()

# 使用 gtimg 接口，按 stock_symbols.txt 清单每 100 只一批拉取
symbols_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_symbols.txt")
final_data, response_total = _fetch_current_prices_by_gtimg(symbols_path, batch_size=100)
print(f"final_data: \n{final_data}")
print(f"response_total: \n{response_total}")

connection_string = os.getenv("DATABASE_URL")
daily_table_name =  os.getenv("DAILY_TABLE_NAME", "stock_daily")
upload_stock_daily_data_to_postgres(today_date, final_data, connection_string, daily_table_name)
print(f"Uploaded {len(final_data)} stock daily data to PostgreSQL successfully")            
