make_plot_complex(
        x=national_employment["occupation_title"],
        y1=national_employment["2023_employment"],
        y2=national_employment["2022_employment"],
        y3=national_employment["2021_employment"],
        y4=national_employment["2020_employment"],
        y5=national_employment["2019_employment"],
        title="national yearly employment",
        xlabel="occupations",
        ylabel="employment")