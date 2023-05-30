import numpy as np
import pickle
import os


DATASET_DIR = "datasets/"


def generate(dataset_name: str,
             number_of_students: int,
             number_of_projects: int,
             number_of_skills: int):
    """
        This method randomly generates a new dataset based on given sizes.
    :param dataset_name: Name of the dataset directory
    :param number_of_students: Number of students
    :param number_of_projects: Number of projects
    :param number_of_skills: Number of skills
    :return: Nothing
    """

    assert number_of_students > 0 and number_of_projects > 0 and number_of_skills > 0, "Invalid arguments"

    # Create directories

    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    dataset_dir = os.path.join(DATASET_DIR, f"{dataset_name}/")

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # Proficiency Levels

    proficiency_levels = np.random.randint(0, 100, (number_of_students, number_of_skills))

    with open(os.path.join(dataset_dir, "proficiency_levels.pkl"), "wb") as f:
        pickle.dump(proficiency_levels, f)

    # Requirements

    requirements = []

    p = 0

    while p < number_of_projects:
        weights = [np.random.random() for s in range(number_of_skills)]

        threshold = np.random.random()

        project_requirements = [1. if weights[s] >= threshold else 0. for s in range(number_of_skills)]

        # At least one skill should be required.

        if sum(project_requirements) > 0:
            requirements.append(project_requirements)
            p += 1

    with open(os.path.join(dataset_dir, "requirements.pkl"), "wb") as f:
        pickle.dump(np.asarray(requirements), f)


if __name__ == "__main__":
    dataset_name = input("Dataset directory: ")

    number_of_students = int(input("Number of students: "))
    number_of_projects = int(input("Number of projects: "))
    number_of_skills = int(input("Number of skills: "))

    generate(dataset_name, number_of_students, number_of_projects, number_of_skills)
