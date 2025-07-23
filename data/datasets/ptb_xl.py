import ast
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class PTB_XL:
  sampling_frequency = 500
  record_duration = 10
  channels = (
    'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
  )
  # shape: (21799, 5000, 12)

  @staticmethod
  def find_records(data_dir):
    record_list = pd.read_csv(path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    record_names = [path.join(data_dir, filename) for filename in record_list.filename_hr.values]
    return record_names

  # utilities copy-pasted from https://github.com/helme/ecg_ptbxl_benchmarking (516740d)

  @staticmethod
  def load_raw_labels(folder):
    Y = pd.read_csv(path.join(folder, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    return Y

  @staticmethod
  def compute_label_aggregations(df, folder, ctype):
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(path.join(folder, 'scp_statements.csv'), index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

      def aggregate_all_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
          if key in diag_agg_df.index:
            tmp.append(key)
        return list(set(tmp))

      def aggregate_subdiagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
          if key in diag_agg_df.index:
            c = diag_agg_df.loc[key].diagnostic_subclass
            if str(c) != 'nan':
              tmp.append(c)
        return list(set(tmp))

      def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
          if key in diag_agg_df.index:
            c = diag_agg_df.loc[key].diagnostic_class
            if str(c) != 'nan':
              tmp.append(c)
        return list(set(tmp))

      diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
      if ctype == 'diagnostic':
        df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
        df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
      elif ctype == 'subdiagnostic':
        df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
        df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
      elif ctype == 'superdiagnostic':
        df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
        df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
      form_agg_df = aggregation_df[aggregation_df.form == 1.0]

      def aggregate_form(y_dic):
        tmp = []
        for key in y_dic.keys():
          if key in form_agg_df.index:
            c = key
            if str(c) != 'nan':
              tmp.append(c)
        return list(set(tmp))

      df['form'] = df.scp_codes.apply(aggregate_form)
      df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
      rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

      def aggregate_rhythm(y_dic):
        tmp = []
        for key in y_dic.keys():
          if key in rhythm_agg_df.index:
            c = key
            if str(c) != 'nan':
              tmp.append(c)
        return list(set(tmp))

      df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
      df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
      df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

  @staticmethod
  def select_data(XX, YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
      X = XX[YY.diagnostic_len > 0]
      Y = YY[YY.diagnostic_len > 0]
      mlb.fit(Y.diagnostic.values)
      y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
      counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
      counts = counts[counts > min_samples]
      YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
      YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
      X = XX[YY.subdiagnostic_len > 0]
      Y = YY[YY.subdiagnostic_len > 0]
      mlb.fit(Y.subdiagnostic.values)
      y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
      counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
      counts = counts[counts > min_samples]
      YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
      YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
      X = XX[YY.superdiagnostic_len > 0]
      Y = YY[YY.superdiagnostic_len > 0]
      mlb.fit(Y.superdiagnostic.values)
      y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
      # filter
      counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
      counts = counts[counts > min_samples]
      YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
      YY['form_len'] = YY.form.apply(lambda x: len(x))
      # select
      X = XX[YY.form_len > 0]
      Y = YY[YY.form_len > 0]
      mlb.fit(Y.form.values)
      y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
      # filter
      counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
      counts = counts[counts > min_samples]
      YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
      YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
      # select
      X = XX[YY.rhythm_len > 0]
      Y = YY[YY.rhythm_len > 0]
      mlb.fit(Y.rhythm.values)
      y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
      # filter
      counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
      counts = counts[counts > min_samples]
      YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
      YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
      # select
      X = XX[YY.all_scp_len > 0]
      Y = YY[YY.all_scp_len > 0]
      mlb.fit(Y.all_scp.values)
      y = mlb.transform(Y.all_scp.values)
    else:
      pass

    # save LabelBinarizer -- commented out for simplicity (also removed `outputfolder` from arguments)
    # with open(outputfolder + 'mlb.pkl', 'wb') as tokenizer:
    #   pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb
