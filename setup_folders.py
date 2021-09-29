from pathlib import Path
import os

def get_current_year():
    from datetime import datetime
    today = datetime.today()
    datem = datetime(today.year, today.month, 1)
    return datem.year

def setup_folders():
    years = [i for i in range(2014, get_current_year()+1)]
    folder_names = ["chromedrivers", 
                    "chromedrivers/1", 
                    "chromedrivers/2", 
                    "chromedrivers/3", 
                    "csv's",
                    "csv's/dkdata"
                    ]
    for i in years:
        folder_names.append(f"csv's/{i}")

    for folder in folder_names:
        _file = Path(f'./{folder}')
        if _file.exists():
            pass
        else:
            os.mkdir(f'./{folder}')

if __name__ == "__main__":
    setup_folders()