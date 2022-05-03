from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def get_lines(filename):

  with open(filename, "r") as f:

    return f.readlines()


def preprocess_data(filename):
    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:

        if line.startswith("###"):

            abstract_lines = ""  # reset the abstract line

        elif line.isspace():

            abstract_line_split = abstract_lines.splitlines()  # splitlines(): Split a string into a list where each line is a list item:

            # Iterate through each line
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}

                target_text_split = abstract_line.split("\t")  # Split target label from text

                line_data["target"] = target_text_split[0]  # get the target label
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)

        else:

            abstract_lines += line

    return abstract_samples


def encode_labels(train_labels, val_labels):

  le = LabelEncoder()

  train_labels_encoded = le.fit_transform(train_labels)
  val_labels_encoded = le.transform(val_labels)

  return train_labels_encoded, val_labels_encoded


def one_hot_labels(train_df, val_df):

  one_hot = OneHotEncoder(sparse=False)

  train_labels_one_hot = one_hot.fit_transform(train_df["target"].to_numpy().reshape(-1,1))
  val_labels_one_hot = one_hot.transform(val_df["target"].to_numpy().reshape(-1,1))

  return train_labels_one_hot, val_labels_one_hot