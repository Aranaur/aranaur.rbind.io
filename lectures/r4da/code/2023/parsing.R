library(pacman)

p_load(tidyverse, rvest,
       janitor, hrbrthemes, jsonlite)

# Wiki -------------------------------------------------------------------------

m100 <- read_html("https://en.wikipedia.org/wiki/Men's_100_metres_world_record_progression")

m100

# Unofficial progression before the IAAF

## Selector
m100 %>% 
  html_element("div+ .wikitable :nth-child(1)") %>% 
  html_table()

## Inspect
pre_iaaf <- m100 %>% 
  html_element("#mw-content-text > div.mw-parser-output > table:nth-child(11)") %>% 
  html_table() %>% 
  clean_names() %>% 
  mutate(date = mdy(date))

pre_iaaf

# Records 1912–1976

iaaf_12_76 <- m100 %>% 
  html_element("#mw-content-text > div.mw-parser-output > table:nth-child(17)") %>% 
  html_table() %>% 
  clean_names() %>% 
  mutate(date = mdy(date))

iaaf_12_76

# Records since 1977

iaaf_77 <- m100 %>% 
  html_element("#mw-content-text > div.mw-parser-output > table:nth-child(23)") %>% 
  html_table() %>% 
  clean_names() %>% 
  mutate(date = mdy(date))

iaaf_77

# Low-altitude record progression 1968–1987

iaaf_68_87 <- m100 %>% 
  html_element("#mw-content-text > div.mw-parser-output > table:nth-child(27)") %>% 
  html_table() %>% 
  clean_names() %>% 
  mutate(date = mdy(date),
         athlete = str_replace(athlete, "\\[.+\\]", ""))

iaaf_68_87

# Об'єднання таблиць

mwr100 <- 
  bind_rows(
    "Pre-IAAF" = pre_iaaf,
    "Pre-automatic" = iaaf_12_76,
    "Modern" = iaaf_77,
    "Low-altitude" = iaaf_68_87,
    .id = 'era'
    ) %>% 
  select(era, time, athlete, nationality, date)

mwr100

mwr100 %>% 
  count(nationality, sort = TRUE)

mwr100_plot <- mwr100 %>% 
  ggplot(aes(date, time, color = fct_reorder2(era, date, time))) +
  geom_point(alpha = 0.7) +
  labs(
    title = "Світові рекорди стометрівки серед чоловіків",
    x = "Дата",
    y = "Час",
    caption = "Джерело: Wikipedia",
    color = "Період"
  ) +
  theme_bw()

plotly::ggplotly(mwr100_plot)
