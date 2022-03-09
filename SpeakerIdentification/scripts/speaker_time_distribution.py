"""
Draw speaker time distribution charts
"""
import json
import os
from datetime import datetime

from pyecharts import options as opts
from pyecharts.charts import Pie, Bar, Page

Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


def visualization():
    colors = ['#c23531', '#2f4554', '#61a0a8', '#d48265', '#749f83', '#ca8622', '#bda29a', '#6e7074', '#546570',
              '#c4ccd3', '#f05b72', '#ef5b9c', '#f47920', '#905a3d', '#fab27b', '#2a5caa', '#444693', '#726930',
              '#b2d235', '#6d8346', '#ac6767', '#1d953f', '#6950a1', '#918597']
    # with open(Root_Dir + '/experiment/speaker_id_dict.json') as f:
    #     speaker_id_dict = json.load(f)
    #
    # print(speaker_id_dict)

    log_dir = Root_Dir + '/experiment/logs/'
    log_files = os.listdir(log_dir)

    # key_list = list(speaker_id_dict.keys())
    # val_list = list(speaker_id_dict.values())
    # bar_number = len(key_list)

    for log_file in log_files:
        # noted, dont use shallow copying by multiplying empty list
        # bars = [[] for i in range(bar_number)]
        bars = {}
        bar_number = 0
        speaker_dist_dict = {}

        # initialize
        # for speaker in val_list:
        #     speaker_dist_dict[speaker] = 0

        log_path = os.path.join(log_dir, log_file)
        with open(log_path, 'r') as f:
            lines = f.readlines()

        l = len(lines)
        start_time = datetime.strptime(lines[1].strip().split('\t')[2][:-7], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(lines[l - 1].strip().split('\t')[2][:-7], '%Y-%m-%d %H:%M:%S')
        last_time = end_time - start_time
        total_seconds = last_time.total_seconds()

        x_bar = []
        count = 0
        for i in range(1, l):

            speaker = lines[i].strip().split('\t')[1]

            if speaker not in speaker_dist_dict:
                speaker_dist_dict[speaker] = 0
                bars[speaker] = [] if count == 0 else [None for i in range(count)]
                # bars[speaker].append(1)
                bar_number += 1

            count += 1


            # position = val_list.index(speaker)
            # key = int(key_list[position])
            time = datetime.strptime(lines[i].strip().split('\t')[2][:-7], '%Y-%m-%d %H:%M:%S')
            period = time - start_time
            x_bar.append(str(period))

            # for j in range(0, bar_number):
            #     bars[j].append(1) if key == j else bars[j].append(None)

            for key in bars.keys():
                bars[key].append(1) if key == speaker else bars[key].append(0)

            speaker_dist_dict[speaker] += 1

        labels = list(speaker_dist_dict.keys())
        distributions = list(speaker_dist_dict.values())
        norm_distributions = [round(float(i) / sum(distributions), 2) for i in distributions]
        seconds_distribution = [int(i * total_seconds) for i in norm_distributions]

        bar = (
            Bar(init_opts=opts.InitOpts(width="1600px", height="200px"))
                .add_xaxis(x_bar)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="Speaker Alignment", pos_top=0, pos_left='center'),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(name="time", boundary_gap=False),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(formatter='{value}'),
                    interval=1, ),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                legend_opts=opts.LegendOpts(pos_top="13%"),
            )

        )

        # for i in range(0, bar_number):

        color_index = -1
        for key in bars.keys():
            color_index += 1
            # bar_data = bars[i]
            bar_data = bars[key]
            name = key
            # name = speaker_id_dict[str(i)]

            bar.add_yaxis(series_name=name, y_axis=bar_data, category_gap=0,
                          label_opts=opts.LabelOpts(is_show=False),
                          itemstyle_opts=opts.ItemStyleOpts(color=colors[color_index]), )

        pie = Pie(init_opts=opts.InitOpts())

        # add data
        pie.add("",
                [list(z) for z in zip(labels, seconds_distribution)],
                label_opts=opts.LabelOpts(
                    position="outside",
                    formatter="{b|{b}: }{c}  {per|{d}%}",
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

        # pie_color = [colors[val_list.index(label)] for label in labels]
        pie_color = [colors[i] for i in range(len(speaker_dist_dict))]
        pie.set_colors(pie_color)

        pie.set_global_opts(
            title_opts=opts.TitleOpts(title="Speaker Time Distribution (seconds)", pos_top=0, pos_left='center'),
            legend_opts=opts.LegendOpts(pos_top=20)
        )

        if not os.path.exists(Root_Dir + "/experiment/charts/"):
            os.mkdir(Root_Dir + "/experiment/charts/")

        page = (
            Page()
            .add(bar, pie)
            .render(Root_Dir + '/experiment/charts/' + str(log_file)[:-4] + '.html')
        )


if __name__ == '__main__':
    visualization()
