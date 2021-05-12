import numpy as np

import bokeh.plotting
from bokeh.models import Range1d, LinearAxis, Div, HoverTool
from bokeh.io import show
from bokeh.io import output_notebook, reset_output, output_file
from bokeh.palettes import d3

from utils import get_naive_datetime_from_datetime


def make_y_limit(y_array, upper_ratio=0.1, lowwer_ratio=0.1):
    min_value = np.amin(y_array)
    max_value = np.amax(y_array)
    diff = max_value - min_value
    return min_value-lowwer_ratio*diff, max_value+upper_ratio*diff


def make_y_limit_multi(y_arrays, upper_ratio=0.1, lowwer_ratio=0.1):
    min_values = []
    max_values = []
    for y_array in y_arrays:
        min_values.append(np.amin(y_array))
        max_values.append(np.amax(y_array))
        
    min_value = min(min_values)
    max_value = max(max_values)
    diff = max_value - min_value
    
    return min_value-lowwer_ratio*diff, max_value+upper_ratio*diff


def make_ticker_text(ticker_value_array, ticker_names):
    div_text = ""
    text_sum_line = 150
    text_sum_count = 0

    for i, ticker_name in enumerate(ticker_names):
        div_text += ticker_name + "="
        text_sum_count += len(ticker_name)
        ticke_value_str = str(ticker_value_array[i])
        div_text += ticke_value_str
        text_sum_count += len(ticke_value_str)

        div_text += ", "
        text_sum_count += 2

        if text_sum_count > text_sum_line:
            div_text += "\n"
            text_sum_count = 0
            
    return div_text


def visualize_portfolio_transform_bokeh(portfolio_state_list, save_path=None, is_save=False, is_show=True, is_jupyter=True):
    # テータの取り出し
    ticker_names = portfolio_state_list[0].names
    colors = d3["Category20"][len(ticker_names)]

    all_price_array = np.stack([one_state.now_price_array for one_state in portfolio_state_list], axis=1)
    all_portfolio_vector = np.stack([one_state.portfolio_vector for one_state in portfolio_state_list], axis=1)
    all_mean_cost_price_array = np.stack([one_state.mean_cost_price_array for one_state in portfolio_state_list], axis=1)
    all_assets_array = np.array([one_state.all_assets for one_state in portfolio_state_list])
    all_datetime_array = np.array([get_naive_datetime_from_datetime(one_state.datetime) for one_state in portfolio_state_list])
    x = np.arange(0, len(portfolio_state_list))


    # sorceの作成
    portfolio_vector_source = {"x":x, "datetime":all_datetime_array}
    price_source_x = []
    price_source_y = []

    mean_cost_price_source_x = []
    mean_cost_price_source_y = []

    for i, ticker_name in enumerate(ticker_names):
        portfolio_vector_source[ticker_name] = all_portfolio_vector[i,:]

        price_source_x.append(x)
        price_source_y.append(all_price_array[i,:]/all_price_array[i,0])

        mean_cost_price_source_x.append(x)
        mean_cost_price_source_y.append(all_mean_cost_price_array[i,:]/all_mean_cost_price_array[i,0])

    # ホバーツールの設定
    #tool_tips = [("x", "@x")]
    tool_tips = [("datetime", "@datetime{%F %H:%M:%S}")]
    tool_tips.extend([(ticker_name, "@"+ticker_name+"{0.000}") for ticker_name in ticker_names])

    hover_tool = HoverTool(
        tooltips=tool_tips,
        formatters={'@datetime' : 'datetime'}
    )

    # 描画

    p1_text = Div(text=make_ticker_text(all_price_array[:,0], ticker_names))

    p1 = bokeh.plotting.figure(plot_width=1200,plot_height=500,title="正規化価格・ポートフォリオ")
    p1.add_tools(hover_tool)

    p1.extra_y_ranges = {"portfolio_vector": Range1d(start=0, end=3)}
    p1.add_layout(LinearAxis(y_range_name="portfolio_vector"), 'right')
    p1.vbar_stack(ticker_names, x='x', width=1, color=colors,y_range_name="portfolio_vector", source=portfolio_vector_source, legend_label=ticker_names, alpha=0.8)

    p1.multi_line(xs=price_source_x, ys=price_source_y, line_color=colors, line_width=2)
    y_min, y_max = make_y_limit_multi(price_source_y, lowwer_ratio=0.1, upper_ratio=0.1)
    y_min -= (y_max - y_min) * 0.66  #  ポートフォリオ割合のためのオフセット
    p1.y_range = Range1d(start=y_min, end=y_max)

    p1.yaxis[0].axis_label = "正規化価格"
    p1.yaxis[1].axis_label = "保有割合"

    p1.xaxis.major_label_overrides = {str(one_x) : str(all_datetime_array[i]) for i, one_x in enumerate(x)}

    p2_text = Div(text=make_ticker_text(all_mean_cost_price_array[:,0], ticker_names))

    p2 = bokeh.plotting.figure(plot_width=1200,plot_height=300,title="正規化平均取得価格・全資産")
    p2.multi_line(xs=mean_cost_price_source_x, ys=mean_cost_price_source_y, line_color=colors, line_width=2)
    y_min, y_max = make_y_limit_multi(mean_cost_price_source_y, lowwer_ratio=0.1, upper_ratio=0.1)
    p2.y_range = Range1d(start=y_min, end=y_max)

    y_max, y_min = make_y_limit(all_assets_array, upper_ratio=0.1, lowwer_ratio=0.1)
    p2.extra_y_ranges = {"all_assets": Range1d(start=y_max, end=y_min)}
    p2.add_layout(LinearAxis(y_range_name="all_assets"), 'right')
    p2.line(x, all_assets_array, color="red", legend_label="all_assets", line_width=4, y_range_name="all_assets")

    # 疑似的なレジェンドをつける
    for ticker_name, color in zip(ticker_names, colors):
        p2.line([], [], legend_label=ticker_name, color=color, line_width=2)

    p2.yaxis[0].axis_label = "正規化平均取得価格"
    p2.yaxis[1].axis_label = "全資産 [円]"

    p2.xaxis.major_label_overrides = {str(one_x) : str(all_datetime_array[i]) for i, one_x in enumerate(x)}

    layout_list = [p1_text, p1, p2_text, p2]
    created_figure = bokeh.layouts.column(*layout_list)

    if is_save:
            if save_path.suffix == ".png":
                bokeh.io.export_png(created_figure, filename=save_path)
            elif save_path.suffix == ".html":
                output_file(save_path)
                bokeh.io.save(created_figure, filename=save_path, title="trading process")    
            else:
                raise Exception("The suffix of save_path is must be '.png' or '.html'.")
            
            return None
    if is_show:
        try:
            reset_output()
            if is_jupyter:
                output_notebook()
            show(created_figure)
        except:
            if is_jupyter:
                output_notebook()
            show(created_figure)
            
        return None
        
    if not is_save and not is_show:
        return layout_list


def visualize_portfolio_rl_bokeh(portfolio_state_list, reward_list, save_path=None, is_save=False, is_show=True, is_jupyter=True):
    all_datetime_array = np.array([get_naive_datetime_from_datetime(one_state.datetime) for one_state in portfolio_state_list])
    reward_array = np.array(reward_list)
    x = np.arange(0, len(portfolio_state_list))

    layout_list = visualize_portfolio_transform_bokeh(portfolio_state_list, is_save=False, is_show=False)

    add_p1 = bokeh.plotting.figure(plot_width=1200,plot_height=300,title="報酬")
    add_p1.line(x, reward_array, legend_label="reward", line_width=2, color="green")
    add_p1.xaxis.major_label_overrides = {str(one_x) : str(all_datetime_array[i]) for i, one_x in enumerate(x)}

    add_p1.yaxis[0].axis_label = "報酬"

    layout_list.extend([add_p1])
    created_figure = bokeh.layouts.column(*layout_list)

    if is_save:
            if save_path.suffix == ".png":
                bokeh.io.export_png(created_figure, filename=save_path)
            elif save_path.suffix == ".html":
                output_file(save_path)
                bokeh.io.save(created_figure, filename=save_path, title="trading process")    
            else:
                raise Exception("The suffix of save_path is must be '.png' or '.html'.")
            
            return None
    if is_show:
        try:
            reset_output()
            if is_jupyter:
                output_notebook()
            show(created_figure)
        except:
            if is_jupyter:
                output_notebook()
            show(created_figure)
            
        return None
        
    if not is_save and not is_show:
        return layout_list

if __name__ == "__main__":
    pass