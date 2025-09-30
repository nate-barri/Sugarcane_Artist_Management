CREATE TABLE analytics_cleaned (
    post_id INT PRIMARY KEY,
    page_id INT,
    page_name TEXT,
    title TEXT,
    description TEXT,
    duration INT,
    publish_year INT,
    publish_month INT,
    publish_day INT,
    publish_time TIME,
    caption_type TEXT,
    permalink TEXT,
    is_crosspost BOOLEAN,
    is_share BOOLEAN,
    post_type TEXT,
    languages TEXT,
    custom_labels TEXT,
    funded_content_status TEXT,
    data_comment TEXT,
    date VARCHAR(90),
    impressions INT,
    reach INT,
    reactions_comments_shares INT,
    reactions INT,
    comments INT,
    shares INT,
    total_clicks INT,
    other_clicks INT,
    matched_audience_targeting_consumption INT,
    link_clicks INT,
    views INT,
    reels_plays_count INT,
    seconds_viewed INT,
    avg_seconds_viewed FLOAT,
    estimated_earnings FLOAT
);
ALTER TABLE analytics_cleaned 
ALTER COLUMN post_id SET DATA TYPE BIGINT USING post_id::BIGINT,
ALTER COLUMN page_id SET DATA TYPE BIGINT USING page_id::BIGINT;


select* from public.analytics_cleaned
truncate table public.analytics_cleaned