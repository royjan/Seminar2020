class GraphUtils:
    @staticmethod
    def create_graph(results: list):
        """
        :param results: a list of learning results from ThreadManager
        """
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib
        from Algorithm import Model
        matplotlib.rcParams['toolbar'] = 'None'  # disable toolbar
        fig = plt.figure(figsize=(32, 12))
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rc('font', **font)
        fig.canvas.set_window_title('Results')
        y_label = [item.score for item in results]
        x_label = np.arange(len(y_label))
        models_labels = [Model.get_model_name(clf) for clf in results]
        models_labels = GraphUtils.get_counts_in_names(models_labels)
        models_labels = [GraphUtils.split_capitalize_words(word) for word in models_labels]
        plt.bar(x_label, np.array(y_label) / 1e9)
        plt.xticks(range(len(models_labels)), models_labels)
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("Results")
        plt.show()

    @staticmethod
    def split_capitalize_words(word: str) -> str:
        """
        :param word: str word
        :return: split lines with capitalize words
        """
        import re
        return re.sub(r"(\w)([A-Z])", r"\1\n\2", word) if not word.isupper() else word

    @staticmethod
    def get_counts_in_names(lst : list) -> list:
        """
        add numbers if duplicates (like: SVC1 SVC2)
        """
        return list(map(
            lambda x: x[1] + str(lst[:x[0]].count(x[1]) + 1) if lst.count(x[1]) > 1 else x[1],
            enumerate(lst)))