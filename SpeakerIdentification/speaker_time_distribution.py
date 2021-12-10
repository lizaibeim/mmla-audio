"""
Draw the pie chart of speaker distribution
"""
import os
from datetime import datetime
from pyecharts import options as opts
from pyecharts.charts import Pie


def draw_pie_chart():
    log_dir = './experiment/logs/'
    log_files = os.listdir(log_dir)

    for log_file in log_files:
        speaker_dist_dict = {}
        log_path = os.path.join(log_dir, log_file)
        with open(log_path, 'r') as f:
            lines = f.readlines()

        l = len(lines)
        start_time = datetime.strptime(lines[1].strip().split('\t')[2][:-7], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(lines[l - 1].strip().split('\t')[2][:-7], '%Y-%m-%d %H:%M:%S')
        last_time = end_time - start_time
        total_seconds = last_time.total_seconds()

        for i in range(1, l):
            speaker = lines[i].strip().split('\t')[1]
            if speaker in speaker_dist_dict.keys():
                speaker_dist_dict[speaker] = speaker_dist_dict[speaker] + 1
            else:
                speaker_dist_dict[speaker] = 1

        labels = list(speaker_dist_dict.keys())
        distributions = list(speaker_dist_dict.values())
        norm_distributions = [round(float(i) / sum(distributions), 2) for i in distributions]
        seconds_distribution = [int(i * total_seconds) for i in norm_distributions]

        pie = Pie(init_opts=opts.InitOpts())
        # add data
        pie.add("",
                [list(z) for z in zip(labels, seconds_distribution)],
                label_opts=opts.LabelOpts(
                    position="outside",
                    formatter="{b|{b}: }{c}  {per|{d}%}",
                    # background_color="#eee",
                    # border_color="#aaa",
                    border_width=0,
                    border_radius=0,
                    rich={
                        "b": {"fontSize": 12, "lineHeight": 20},
                        "per": {
                            "color": "#eee",
                            "backgroundColor": "#334455",
                            "padding": [1, 1],
                            "borderRadius": 0,
                        },
                    },
                ),

                )

        pie.set_global_opts(
            title_opts=opts.TitleOpts(title="Speaker Time Distribution (seconds)", pos_top=0, pos_left='center'),
            legend_opts=opts.LegendOpts(pos_bottom=0)
        )
        # pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))

        pie.render('./experiment/pie_charts/' + str(log_file) + '.html')


if __name__ == '__main__':
    draw_pie_chart()
