import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import keras

import os

from student_utils import create_tf_numeric_feature


def aggregate_dataset(df, grouping_field_list, array_field):
    # get dummies df
    dummy_df = df.set_index("encounter_id")[array_field].str.get_dummies()
    dummy_col_list = list(dummy_df.columns)

    agg_dummy_df = (
        dummy_df.reset_index()
        .groupby("encounter_id")
        .apply(max)
        .drop("encounter_id", axis=1)
        .reset_index()
    )
    no_dupe_df = df[grouping_field_list].drop_duplicates()
    merged_df = pd.merge(no_dupe_df, agg_dummy_df, on="encounter_id")

    return merged_df, dummy_col_list


def cast_df(df, col, d_type=str):
    return df[col].astype(d_type)


def impute_df(df, col, impute_value=0):
    return df[col].fillna(impute_value)


def preprocess_df(
    df,
    categorical_col_list,
    numerical_col_list,
    predictor,
    numerical_impute_value=0,
):
    processed_df = df.copy()
    processed_df[predictor] = processed_df[predictor].astype(float)
    for c in categorical_col_list:
        processed_df[c] = processed_df[c].astype(str)
    for c in numerical_col_list:
        processed_df[c].fillna(numerical_impute_value, inplace=True)
    return processed_df


def show_group_stats_viz(df, group):
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind="barh"))


# adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor, batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


# build vocab for categorical features
def write_vocabulary_file(
    vocab_list, field_name, default_value, vocab_dir="./diabetes_vocab/"
):
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    # put default value in first row as TF requires
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0)
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path


def build_vocab_files(df, categorical_column_list, default_value="00"):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list


"""
Adapted from Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb
"""


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def demo(feature_column, example_batch):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)


def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()["mean"]
    std = df[col].describe()["std"]
    return mean, std


def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list
