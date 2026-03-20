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