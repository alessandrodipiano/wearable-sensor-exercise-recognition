import pandas as pd 

def build_df(root):

    templates = []

    label_map = {
        1: "correct",
        2: "fast",
        3: "low_amplitude"
    }

    sample_id = 0

    for subject_dir in root.glob("s*"):
        for exercise_dir in subject_dir.glob("e*"):
            times_path = exercise_dir / "template_times.txt"
            

            template_times = pd.read_csv(times_path, sep=";")

            for unit_dir in exercise_dir.glob("u*"):
                file_path = unit_dir / "template_session.txt"
                

                df = pd.read_csv(file_path, sep=";")
                df["subject"] = subject_dir.name
                df["exercise"] = exercise_dir.name
                df["unit"] = unit_dir.name
            

                for _, row in template_times.iterrows():
                    start = row["start"]   
                    end = row["end"]
                    exec_type = row["execution type"]

                    segment = df[
                        (df["time index"] >= start) &
                        (df["time index"] <= end)
                    ].copy()

                    segment["execution_type"] = label_map[exec_type]
                    segment["sample_id"] = sample_id

                    templates.append(segment)

                    sample_id += 1   
    templates_df = pd.concat(templates, ignore_index=True)

    return templates_df






 
def analyze_segment_lengths(templates_df):
    """
    For each sample_id, count the number of timesteps.
    Then summarize by execution_type: mean, std, min, max.
    """
 
    # Step 1: count timesteps per sample
    lengths = (
        templates_df
        .groupby([ "execution_type", "subject", "exercise"])
        .size()
        .reset_index(name="n_timesteps")
    )
 
    # Step 2: summary stats grouped by execution type
    summary = (
        lengths
        .groupby("execution_type")["n_timesteps"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(1)
    )
 
    print("=== Timesteps per execution type ===")
    print(summary)
    print()
 
    # Step 3: per exercise breakdown (to see if pattern holds across all 8)
    per_exercise_mean = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .mean()
        .round(1)
        .unstack("execution_type")
    )
    per_exercise_mean.columns = [f"{c}_mean" for c in per_exercise_mean.columns]
 
    per_exercise_max = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .max()
        .unstack("execution_type")
    )
    per_exercise_max.columns = [f"{c}_max" for c in per_exercise_max.columns]
 
    per_exercise_min = (
        lengths
        .groupby(["exercise", "execution_type"])["n_timesteps"]
        .min()
        .unstack("execution_type")
    )
    per_exercise_min.columns = [f"{c}_min" for c in per_exercise_min.columns]
 
    per_exercise = pd.concat([per_exercise_mean, per_exercise_max, per_exercise_min], axis=1).sort_index(axis=1)
 
    print("=== Mean, min and max timesteps per exercise × execution type ===")
    print(per_exercise)
 
    return lengths, summary, per_exercise






def compare_sensor_means(templates_df, exec_types=("correct", "low_amplitude")):
    """
    Compare average sensor readings between two execution types,
    broken down by exercise × unit combination.
 
    For each (exercise, unit) pair and each sensor group (acc, gyr, mag),
    computes the mean per axis per execution type, plus the diff row.
 
    Returns a dict: { (exercise, unit): { sensor: DataFrame } }
    """
 
    sensor_cols = {
        "acc": ["acc_x", "acc_y", "acc_z"],
        "gyr": ["gyr_x", "gyr_y", "gyr_z"],
        "mag": ["mag_x", "mag_y", "mag_z"],
    }
 
    subset = templates_df[templates_df["execution_type"].isin(exec_types)]
 
    results = {}
 
    for (exercise, unit), group in subset.groupby(["exercise", "unit"]):
        key = (exercise, unit)
        results[key] = {}
 
        print(f"\n{'='*50}")
        print(f"Exercise: {exercise}  |  Unit: {unit}")
        print(f"{'='*50}")
 
        for sensor, cols in sensor_cols.items():
            sensor_means = (
                group
                .groupby("execution_type")[cols]
                .mean()
                .round(4)
            )
 
            # diff row: second exec_type minus first
            if all(t in sensor_means.index for t in exec_types):
                sensor_means.loc[f"diff ({exec_types[1]} - {exec_types[0]})"] = (
                    sensor_means.loc[exec_types[1]] - sensor_means.loc[exec_types[0]]
                )
 
            results[key][sensor] = sensor_means
 
            print(f"\n  {sensor.upper()}")
            print(sensor_means.to_string())
 
    return results