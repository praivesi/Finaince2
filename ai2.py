import FinanceDataReader as fdr
import datetime

start_date = datetime.datetime.now() - datetime.timedelta(days=365*20)
# 현재 시간으로부터 20년 전 날짜 계산


df_kospi = fdr.DataReader('KS11', start_date).add_suffix('_kospi')
df_kosdaq = fdr.DataReader('KQ11', start_date).add_suffix('_kosdaq')
df_dow = fdr.DataReader('DJI', start_date).add_suffix('_dow')
df_nasdaq = fdr.DataReader('IXIC', start_date).add_suffix('_nasdaq')
df_snp500 = fdr.DataReader('US500', start_date).add_suffix('_s&p500')