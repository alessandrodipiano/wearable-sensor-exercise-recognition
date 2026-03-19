import pandas as pd 

def build_df(root):

    templates = []

    label_map = {
        1: "correct",
        2: "fast",
        3: "low_amplitude"
    }

    for subject_dir in root.glob("s*"):
        for exercise_dir in subject_dir.glob("e*"):
            times_path = exercise_dir / "template_times.txt"
            if not times_path.exists():
                continue

            template_times = pd.read_csv(times_path, sep=";")

            for unit_dir in exercise_dir.glob("u*"):
                file_path = unit_dir / "template_session.txt"
                if not file_path.exists():
                    continue

                df = pd.read_csv(file_path, sep=";")
                df["subject"] = subject_dir.name
                df["exercise"] = exercise_dir.name
                df["unit"] = unit_dir.name
            

                for _, row in template_times.iterrows():
                    start = row["start"]   # replace with actual column names
                    end = row["end"]
                    exec_type = row["execution type"]

                    segment = df[(df["time index"] >= start) & (df["time index"] <= end)].copy()
                    segment["execution_type"] = label_map[exec_type]
                    templates.append(segment) 
    templates_df = pd.concat(templates, ignore_index=True)

    return templates_df