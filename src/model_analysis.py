
__author__ = "Jacob Taylor Cassady"
__email__ = "jcassady@jh.edu"

# Built-in Libraries
from enum import Enum
from typing import Dict, List, Tuple
from yaml import dump as yaml_dump

# External Libraries
from numpy import array, linspace, average, std
from pandas import DataFrame
from matplotlib.pyplot import subplots, subplots_adjust, show, clf
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Local Libraries
from utils import load_adfes_dataframe


class TARGET_OF_INTEREST(Enum):
    GEOGRAPHIC_TAG: str = 'geographic_tag'
    GENDER: str = 'gender'
    MODEL: str = 'model_id'
    EMOTION: str = 'emotion'

    def __str__(self):
        return self.value
    
    def get_eigenface_count(self) -> int:
        eigenface_count: Dict[TARGET_OF_INTEREST, int] = {
            TARGET_OF_INTEREST.GEOGRAPHIC_TAG: 50,
            TARGET_OF_INTEREST.GENDER: 65,
            TARGET_OF_INTEREST.MODEL: 50,
            TARGET_OF_INTEREST.EMOTION: 75
        }
        
        if self in eigenface_count:
            return eigenface_count[self]
        else:
            raise NotImplementedError

    def get_confusion_matrix_display_count(self):
        display_count: Dict[TARGET_OF_INTEREST, int] = {
            TARGET_OF_INTEREST.GEOGRAPHIC_TAG: 2,
            TARGET_OF_INTEREST.GENDER: 2,
            TARGET_OF_INTEREST.MODEL: 20,
            TARGET_OF_INTEREST.EMOTION: 10
        }

        if self in display_count:
            return display_count[self]
        else:
            raise NotImplementedError
    
    def get_subplots_dimension(self):
        subplots_dimensions: Dict[TARGET_OF_INTEREST, Tuple[int, int]] = {
            TARGET_OF_INTEREST.GEOGRAPHIC_TAG: (1, 2),
            TARGET_OF_INTEREST.GENDER: (1, 2),
            TARGET_OF_INTEREST.MODEL: (4, 5),
            TARGET_OF_INTEREST.EMOTION: (2, 5)
        }

        if self in subplots_dimensions:
            return subplots_dimensions[self]
        else:
            raise NotImplementedError


def train_test_on_all_targets(test_count: int = 5):
    df: DataFrame = load_adfes_dataframe()

    for target in TARGET_OF_INTEREST:
        target_label: str = str(target)

        image_data: array = array(df['image'].apply(lambda img: array(img).flatten()).to_list())
        targets: array = array(df[target_label].to_list()).reshape(-1, 1)
        
        # Convert labels to integers
        label_encoder = LabelEncoder()
        targets: array = label_encoder.fit_transform(targets)

        # Split data into training and testing sets
        best_targets: array = None
        best_predictions: array = None
        best_accuracy: float = 0.0
        for test_index in range(test_count):
            train_features, test_features, train_targets, test_targets = train_test_split(image_data, targets, stratify=targets, test_size=0.2)

            # Perform PCA on the training data to get eigenfaces
            eigenface_count: int = target.get_eigenface_count()
            dataset_pca = PCA(svd_solver='randomized', n_components=eigenface_count, whiten=True)
            train_features_pca = dataset_pca.fit_transform(train_features)
            print(f'explained_variance when using {eigenface_count} eigenfaces: {sum(dataset_pca.explained_variance_ratio_)*100}')

            model = SVC(kernel='linear', C=3.0)
            model.fit(train_features_pca, train_targets)

            # Evaluate the model
            test_features_pca = dataset_pca.transform(test_features)
            print(f'train_features_pca.shape: {train_features_pca.shape}')
            print(f'test_features_pca.shape: {test_features_pca.shape}')
            predictions: array = model.predict(test_features_pca)
            report = classification_report(test_targets, predictions, target_names=label_encoder.classes_, output_dict=True)
            accuracy: float = report['macro avg']['f1-score']

            if best_targets is None:
                best_targets = test_targets
                best_predictions = predictions
                best_accuracy = accuracy
                print(f'{target} - new best\n{report}')
            elif best_accuracy < accuracy:
                best_targets = test_targets
                best_predictions = predictions
                best_accuracy = accuracy
                print(f'{target} - new best\n{report}')

            if accuracy == 1.0:
                print(f'{target} - perfect accuracy after {test_index+1} test(s).')
                break

        # Plot the confusion matrices (https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python)
        with open(f'{target_label}_{eigenface_count}_classification_report.yaml', 'w') as file:
            yaml_dump(report, file)

        subplots_dimensions: Tuple[int, int] = target.get_subplots_dimension()
        f, axes = subplots(subplots_dimensions[0], subplots_dimensions[1])
        axes = axes.ravel()
        for i in range(len(label_encoder.classes_[:target.get_confusion_matrix_display_count()])):
            test_targets_i = best_targets==i
            predictions_i = best_predictions==i

            disp = ConfusionMatrixDisplay(confusion_matrix(test_targets_i,  predictions_i),
                                          display_labels=['F', 'T'])
            disp.plot(ax=axes[i], values_format='.4g', cmap='Blues')

            disp.ax_.set_title(f'{label_encoder.classes_[i]}')
            if i % subplots_dimensions[1] != 0:
                disp.ax_.set_ylabel('')
            if i < target.get_confusion_matrix_display_count()-subplots_dimensions[1]:
                disp.ax_.set_xlabel('')
            disp.im_.colorbar.remove()

        subplots_adjust(wspace=0.10, hspace=0.1)
        show()


def eigenface_graph_targets(maximum_eigenfaces: int = 125, test_count: int = 5):
    df: DataFrame = load_adfes_dataframe()
    eigenface_counts: array = linspace(10, maximum_eigenfaces, 25, dtype=int)

    average_accuracies: Dict[str, List[float]] = {}
    standard_deviations: Dict[str, List[float]] = {}
    explained_variance: Dict[str, List[float]] = {}

    for target in TARGET_OF_INTEREST:
        target_label: str = str(target)
        average_accuracies[target_label] = []
        explained_variance[target_label] = []
        standard_deviations[target_label] = []
        print(f'Performing analysis of eigenface training performance on {target_label} over {eigenface_counts} eigenfaces. {test_count} tests per eigenface count.')

        image_data: array = array(df['image'].apply(lambda img: array(img).flatten()).to_list())
        targets: array = array(df[target_label].to_list()).reshape(-1, 1)
        
        # Convert labels to integers
        label_encoder = LabelEncoder()
        targets: array = label_encoder.fit_transform(targets)

        # Split data into training and testing sets
        for eigenface_count in eigenface_counts:
            accuracies: List[float] = []
            for _ in range(test_count):
                train_features, test_features, train_targets, test_targets = train_test_split(image_data, targets, stratify=targets, test_size=0.2)

                # Perform PCA on the training data to get eigenfaces
                dataset_pca = PCA(svd_solver='randomized', n_components=eigenface_count, whiten=True)
                train_features_pca = dataset_pca.fit_transform(train_features)

                model = SVC(kernel='linear', C=3.0)
                model.fit(train_features_pca, train_targets)

                # Evaluate the model
                predictions: array = model.predict(dataset_pca.transform(test_features))
                report = classification_report(test_targets, predictions, target_names=label_encoder.classes_, output_dict=True)
                accuracy: float = report['macro avg']['f1-score']
                accuracies.append(accuracy)

            average_accuracies[target_label].append(average(accuracies))
            standard_deviations[target_label].append(std(accuracies))
            explained_variance[target_label].append(sum(dataset_pca.explained_variance_ratio_))
        print(f'{target_label} - {eigenface_count} eigenfaces: {average_accuracies[target_label][-1]} +/- {standard_deviations[target_label][-1]}')

    _, axes = subplots(2, 2)
    axes = axes.ravel()
    for target_index, target in enumerate(list(TARGET_OF_INTEREST)):
        target_label: str = str(target)
        targets_average_accuracies: List[float] = average_accuracies[target_label]
        targets_accuracy_std: List[float] = standard_deviations[target_label]
        print(f'{target_label} - accuracies: {type(targets_average_accuracies)} | {targets_average_accuracies}')
        targets_explained_variances: List[float] = explained_variance[target_label]
        print(f'{target_label} - expalined variances: {type(targets_explained_variances)} | {targets_explained_variances}')

        ax2 = axes[target_index].twinx() # Create a second y-axis for the explained variance
        axes[target_index].errorbar(eigenface_counts, array(targets_average_accuracies), yerr=targets_accuracy_std, fmt='o', ecolor='blue')
        ax2.plot(eigenface_counts, array(targets_explained_variances), 'p-', color='purple')

        axes[target_index].set_title(f'{target_label}')
        if target_index % 2 == 0:
            axes[target_index].set_ylabel('average accuracy')
        else:
            ax2.set_ylabel('explained variance')

        if target_index >= 2:
            axes[target_index].set_xlabel('Eigenfaces')

        # Change the color of the left axis to blue and the right axis to purple
        axes[target_index].tick_params(axis='y', colors='blue')
        ax2.tick_params(axis='y', colors='purple')

    subplots_adjust(wspace=0.10, hspace=0.1)
    show()


def eigenface_portraits(eigenface_count=65):
    df: DataFrame = load_adfes_dataframe()

    image_data: array = array(df['image'].apply(lambda img: array(img).flatten()).to_list())
    # Targets are irrelvant but including it to ease the process of splitting the data
    _: array = array(df[str(TARGET_OF_INTEREST.EMOTION)].to_list()).reshape(-1, 1)

    # Split data into training and testing sets
    train_features, _, _, _ = train_test_split(image_data, _, stratify=_, test_size=0.2)

    # Perform PCA on the training data to get eigenfaces
    dataset_pca = PCA(svd_solver='randomized', n_components=eigenface_count, whiten=True).fit(train_features)

    # !! Due to development cost and low time, this section of code is not dynamic. It needs to be updated depending on the number of eigenfaces !!
    # Create a 3x3 grid of subplots to display the first 9 eigenfaces of the training data
    axis_count: int = 9
    _, axes = subplots(3, 3)
    axes = axes.ravel()
    for eigenface_index in range(axis_count):
        eigenface = dataset_pca.components_[eigenface_index].reshape(576, 720, 3)*255.
        axes[eigenface_index].imshow(eigenface)
        axes[eigenface_index].set_title(f'eigenface {eigenface_index+1}')

    # subplots_adjust(wspace=0.10, hspace=0.1)
    show()

    # Create a 3x3 grid of subplots to display the last 9 eigenfaces of the training data
    clf()
    _, axes = subplots(3, 3)
    axes = axes.ravel()
    for axis_index in range(axis_count):
        eigenface_index = len(dataset_pca.components_)-axis_index-1
        eigenface = dataset_pca.components_[eigenface_index].reshape(576, 720, 3)*255
        axes[axis_index].imshow(eigenface)
        axes[axis_index].set_title(f'eigenface {eigenface_index+1}')

    # subplots_adjust(wspace=0.10, hspace=0.1)
    show()


if __name__ == '__main__':
    # eigenface_graph_targets(125, 5)
    train_test_on_all_targets()
    # eigenface_graph_targets()
