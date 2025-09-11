-- 0) Schema + platform dimension
CREATE SCHEMA IF NOT EXISTS dw;

CREATE TABLE IF NOT EXISTS dw.dim_platform (
  platform_sk   smallserial PRIMARY KEY,
  platform_code text UNIQUE NOT NULL,
  platform_name text NOT NULL
);

INSERT INTO dw.dim_platform(platform_code, platform_name)
VALUES ('facebook','Facebook'), ('youtube','YouTube')
ON CONFLICT (platform_code) DO NOTHING;

-- 1) Date dimension
CREATE TABLE IF NOT EXISTS dw.dim_date (
  date_sk          integer PRIMARY KEY,  -- YYYYMMDD
  date_actual      date    NOT NULL,
  year             integer NOT NULL,
  month            integer NOT NULL,
  day              integer NOT NULL,
  dow_iso          integer NOT NULL,     -- 1=Mon..7=Sun
  week_iso         integer NOT NULL,     -- ISO week
  month_name       text    NOT NULL,
  quarter          integer NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_dim_date_date ON dw.dim_date(date_actual);

-- 2) Filler function (safe to call multiple times)
CREATE OR REPLACE FUNCTION dw.fill_dim_date(p_start date, p_end date)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE d date;
BEGIN
  IF p_start IS NULL THEN RETURN; END IF;
  IF p_end   IS NULL THEN p_end := p_start; END IF;
  d := LEAST(p_start, p_end);
  WHILE d <= GREATEST(p_start, p_end) LOOP
    INSERT INTO dw.dim_date(date_sk, date_actual, year, month, day, dow_iso, week_iso, month_name, quarter)
    VALUES (
      (to_char(d,'YYYYMMDD'))::int,
      d,
      EXTRACT(YEAR  FROM d)::int,
      EXTRACT(MONTH FROM d)::int,
      EXTRACT(DAY   FROM d)::int,
      EXTRACT(ISODOW FROM d)::int,
      EXTRACT(WEEK   FROM d)::int,
      to_char(d,'Mon'),
      EXTRACT(QUARTER FROM d)::int
    )
    ON CONFLICT (date_sk) DO NOTHING;
    d := d + INTERVAL '1 day';
  END LOOP;
END;
$$;
----01_facebook_dw_and_automation.sql
-- 1) Dimensions & Fact

-- Page (by platform)
CREATE TABLE IF NOT EXISTS dw.dim_page (
  page_sk      bigserial PRIMARY KEY,
  platform_sk  smallint NOT NULL REFERENCES dw.dim_platform(platform_sk),
  page_id      text     NOT NULL,
  page_name    text,
  CONSTRAINT uq_dim_page UNIQUE (platform_sk, page_id)
);
CREATE INDEX IF NOT EXISTS ix_dim_page_id ON dw.dim_page(page_id);

-- Post (content item)
CREATE TABLE IF NOT EXISTS dw.dim_post (
  post_sk                bigserial PRIMARY KEY,
  platform_sk            smallint NOT NULL REFERENCES dw.dim_platform(platform_sk),
  page_sk                bigint   NOT NULL REFERENCES dw.dim_page(page_sk),
  post_id                text     NOT NULL,
  title                  text,
  description            text,
  post_type              text,
  duration_sec           int,
  published_at           timestamptz,
  permalink              text,
  is_crosspost           boolean,
  is_share               boolean,
  funded_content_status  smallint,
  CONSTRAINT uq_dim_post UNIQUE (platform_sk, post_id)
);
CREATE INDEX IF NOT EXISTS ix_dim_post_id ON dw.dim_post(post_id);

-- Fact (one summary row per post)
CREATE TABLE IF NOT EXISTS dw.fact_facebook_post_summary (
  fact_sk                   bigserial PRIMARY KEY,
  post_sk                   bigint  NOT NULL UNIQUE REFERENCES dw.dim_post(post_sk),
  date_sk_posted            integer REFERENCES dw.dim_date(date_sk),
  reach                     bigint  CHECK (reach >= 0),
  impressions               bigint  CHECK (impressions >= 0),
  shares                    int     CHECK (shares >= 0),
  comments                  int     CHECK (comments >= 0),
  reactions                 int     CHECK (reactions >= 0),
  seconds_viewed            numeric(20,4) CHECK (seconds_viewed >= 0),
  average_seconds_viewed    numeric(12,4) CHECK (average_seconds_viewed >= 0),
  created_at                timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_fb_fact_post ON dw.fact_facebook_post_summary(post_sk);
CREATE INDEX IF NOT EXISTS ix_fb_fact_date ON dw.fact_facebook_post_summary(date_sk_posted);

-- 2) Views (handy for BI)
CREATE OR REPLACE VIEW dw.vw_fb_kpis_total AS
SELECT
  COUNT(*)                              AS posts,
  SUM(f.reach)                          AS reach,
  SUM(f.impressions)                    AS impressions,
  SUM(f.reactions)                      AS reactions,
  SUM(f.comments)                       AS comments,
  SUM(f.shares)                         AS shares,
  SUM(f.seconds_viewed)                 AS seconds_viewed,
  AVG(f.average_seconds_viewed)         AS avg_seconds_viewed
FROM dw.fact_facebook_post_summary f;

CREATE OR REPLACE VIEW dw.vw_fb_by_publish_month AS
SELECT
  date_trunc('month', d.date_actual)::date AS month,
  COUNT(*)        AS posts,
  SUM(f.reach)    AS reach,
  SUM(f.impressions) AS impressions,
  SUM(f.reactions)   AS reactions,
  SUM(f.comments)    AS comments,
  SUM(f.shares)      AS shares
FROM dw.fact_facebook_post_summary f
JOIN dw.dim_date d ON d.date_sk = f.date_sk_posted
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW dw.vw_fb_top_posts AS
SELECT
  dp.post_id,
  dp.title,
  dp.post_type,
  dp.published_at,
  f.impressions,
  f.reach,
  f.reactions, f.comments, f.shares,
  (NULLIF((f.reactions+f.comments+f.shares),0)::numeric / NULLIF(f.impressions,0))::numeric(8,5) AS engagement_rate,
  f.seconds_viewed, f.average_seconds_viewed
FROM dw.fact_facebook_post_summary f
JOIN dw.dim_post dp ON dp.post_sk = f.post_sk
ORDER BY engagement_rate DESC NULLS LAST;

-- 3) Sync routine (landing -> DW), NaN-safe
CREATE OR REPLACE FUNCTION dw.sync_facebook_from_landing()
RETURNS void
LANGUAGE plpgsql
AS $dw$
BEGIN
  -- Ensure dim_date covers posted dates
  PERFORM dw.fill_dim_date(
    (SELECT MIN(COALESCE(make_date(year,month,day), publish_time::date))
     FROM public.facebook_data_set
     WHERE publish_time IS NOT NULL
        OR (year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL)),
    CURRENT_DATE
  );

  -- Upsert pages
  WITH src AS (
    SELECT
      (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='facebook') AS platform_sk,
      NULLIF(TRIM(page_id::text),'') AS page_id,
      page_name
    FROM public.facebook_data_set
    WHERE page_id IS NOT NULL
  ),
  d AS (
    SELECT * FROM (
      SELECT s.*, ROW_NUMBER() OVER (PARTITION BY platform_sk,page_id ORDER BY page_name NULLS LAST) rn
      FROM src s
    ) z WHERE rn=1
  )
  INSERT INTO dw.dim_page(platform_sk, page_id, page_name)
  SELECT platform_sk, page_id, page_name FROM d
  ON CONFLICT (platform_sk, page_id) DO UPDATE
  SET page_name = COALESCE(EXCLUDED.page_name, dw.dim_page.page_name);

  -- Upsert posts
  WITH src AS (
    SELECT
      (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='facebook') AS platform_sk,
      p.page_sk,
      f.post_id::text AS post_id,
      f.title, f.description, f.post_type,
      f.duration_sec AS duration_num,
      COALESCE(
        CASE WHEN f.year IS NOT NULL AND f.month IS NOT NULL AND f.day IS NOT NULL
             THEN make_timestamp(f.year,f.month,f.day,
                     split_part(f.time::text,':',1)::int,
                     split_part(f.time::text,':',2)::int,
                     COALESCE(split_part(f.time::text,':',3)::int,0))
        END,
        f.publish_time
      )::timestamptz AS published_at,
      f.permalink,
      f.is_crosspost,
      f.is_share,
      f.funded_content_status::text AS funded_text
    FROM public.facebook_data_set f
    JOIN dw.dim_page p
      ON p.platform_sk = (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='facebook')
     AND p.page_id = f.page_id::text
    WHERE f.post_id IS NOT NULL AND f.post_id::text <> ''
  ),
  norm AS (
    SELECT
      platform_sk, page_sk, post_id, title, description, post_type,
      CASE WHEN duration_num IS NULL THEN NULL ELSE round(duration_num)::int END AS duration_sec,
      published_at, permalink, is_crosspost, is_share,
      CASE WHEN funded_text ~ '^\d+$' THEN funded_text::int ELSE NULL END AS funded_content_status
    FROM src
  ),
  dd AS (
    SELECT * FROM (
      SELECT n.*, ROW_NUMBER() OVER (
        PARTITION BY platform_sk, post_id
        ORDER BY published_at DESC NULLS LAST, title NULLS LAST
      ) rn
      FROM norm n
    ) z WHERE rn=1
  )
  INSERT INTO dw.dim_post(
    platform_sk, page_sk, post_id, title, description, post_type, duration_sec,
    published_at, permalink, is_crosspost, is_share, funded_content_status
  )
  SELECT platform_sk, page_sk, post_id, title, description, post_type, duration_sec,
         published_at, permalink, is_crosspost, is_share, funded_content_status
  FROM dd
  ON CONFLICT (platform_sk, post_id) DO UPDATE
  SET title                 = COALESCE(EXCLUDED.title, dw.dim_post.title),
      description           = COALESCE(EXCLUDED.description, dw.dim_post.description),
      post_type             = COALESCE(EXCLUDED.post_type, dw.dim_post.post_type),
      duration_sec          = COALESCE(EXCLUDED.duration_sec, dw.dim_post.duration_sec),
      published_at          = COALESCE(EXCLUDED.published_at, dw.dim_post.published_at),
      permalink             = COALESCE(EXCLUDED.permalink, dw.dim_post.permalink),
      is_crosspost          = COALESCE(EXCLUDED.is_crosspost, dw.dim_post.is_crosspost),
      is_share              = COALESCE(EXCLUDED.is_share, dw.dim_post.is_share),
      funded_content_status = COALESCE(EXCLUDED.funded_content_status, dw.dim_post.funded_content_status);

  -- Upsert fact (dedup + NaN/Inf safe)
  WITH s_base AS (
    SELECT
      f.post_id::text AS post_id,
      ROW_NUMBER() OVER (
        PARTITION BY f.post_id::text
        ORDER BY COALESCE(f.reach,0) DESC, COALESCE(f.impressions,0) DESC
      ) rn,
      f.reach, f.impressions, f.shares, f.comments, f.reactions,
      f.seconds_viewed, f.average_seconds_viewed,
      COALESCE(
        CASE WHEN f.year IS NOT NULL AND f.month IS NOT NULL AND f.day IS NOT NULL
             THEN make_date(f.year, f.month, f.day) END,
        f.publish_time::date
      ) AS posted_date
    FROM public.facebook_data_set f
    WHERE f.post_id IS NOT NULL AND f.post_id::text <> ''
  ),
  s_one AS (SELECT * FROM s_base WHERE rn=1),
  s_cast AS (
    SELECT
      post_id, posted_date,
      CASE WHEN reach::text                   ~* '^(nan|inf|-inf)$' THEN NULL ELSE reach END                   AS reach_n,
      CASE WHEN impressions::text             ~* '^(nan|inf|-inf)$' THEN NULL ELSE impressions END             AS impressions_n,
      CASE WHEN shares::text                  ~* '^(nan|inf|-inf)$' THEN NULL ELSE shares END                  AS shares_n,
      CASE WHEN comments::text                ~* '^(nan|inf|-inf)$' THEN NULL ELSE comments END                AS comments_n,
      CASE WHEN reactions::text               ~* '^(nan|inf|-inf)$' THEN NULL ELSE reactions END               AS reactions_n,
      CASE WHEN seconds_viewed::text          ~* '^(nan|inf|-inf)$' THEN NULL ELSE seconds_viewed END          AS seconds_viewed_n,
      CASE WHEN average_seconds_viewed::text  ~* '^(nan|inf|-inf)$' THEN NULL ELSE average_seconds_viewed END  AS avg_sec_viewed_n
    FROM s_one
  )
  INSERT INTO dw.fact_facebook_post_summary(
    post_sk, date_sk_posted, reach, impressions, shares, comments, reactions,
    seconds_viewed, average_seconds_viewed
  )
  SELECT
    dp.post_sk,
    (to_char(c.posted_date,'YYYYMMDD'))::int,
    COALESCE(c.reach_n,0)::bigint,
    COALESCE(c.impressions_n,0)::bigint,
    COALESCE(c.shares_n,0)::int,
    COALESCE(c.comments_n,0)::int,
    COALESCE(c.reactions_n,0)::int,
    COALESCE(c.seconds_viewed_n,0)::numeric(20,4),
    COALESCE(c.avg_sec_viewed_n,0)::numeric(12,4)
  FROM s_cast c
  JOIN dw.dim_post dp
    ON dp.platform_sk = (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='facebook')
   AND dp.post_id = c.post_id
  ON CONFLICT (post_sk) DO UPDATE
  SET reach                  = EXCLUDED.reach,
      impressions            = EXCLUDED.impressions,
      shares                 = EXCLUDED.shares,
      comments               = EXCLUDED.comments,
      reactions              = EXCLUDED.reactions,
      seconds_viewed         = EXCLUDED.seconds_viewed,
      average_seconds_viewed = EXCLUDED.average_seconds_viewed,
      date_sk_posted         = EXCLUDED.date_sk_posted;
END;
$dw$;

-- 4) Trigger function + trigger (fires once per INSERT/UPDATE batch)
CREATE OR REPLACE FUNCTION dw.trg_fb_sync_stmt()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = dw, public
AS $$
BEGIN
  PERFORM dw.sync_facebook_from_landing();
  RETURN NULL;
END
$$;

DROP TRIGGER IF EXISTS fb_sync_stmt_insupd ON public.facebook_data_set;
CREATE TRIGGER fb_sync_stmt_insupd
AFTER INSERT OR UPDATE ON public.facebook_data_set
FOR EACH STATEMENT
EXECUTE FUNCTION dw.trg_fb_sync_stmt();


---02_youtube_dw_and_automation.sql
-- 1) Dimension & Fact

-- Video dimension (note: published_at as DATE; IDs/strings are TEXT to avoid truncation)
CREATE TABLE IF NOT EXISTS dw.dim_video (
  video_sk         bigserial PRIMARY KEY,
  platform_sk      smallint NOT NULL REFERENCES dw.dim_platform(platform_sk),
  youtube_video_id text     NOT NULL,
  title            text,
  published_at     date,
  duration_sec     integer,
  url              text,
  category         text,
  CONSTRAINT uq_dim_video UNIQUE (platform_sk, youtube_video_id)
);

-- Fact (one summary row per video)
CREATE TABLE IF NOT EXISTS dw.fact_youtube_video_summary (
  fact_sk                bigserial PRIMARY KEY,
  video_sk               bigint  NOT NULL UNIQUE REFERENCES dw.dim_video(video_sk),
  impressions            bigint  CHECK (impressions >= 0),
  impressions_ctr_pct    numeric CHECK (impressions_ctr_pct >= 0),  -- store 0..100
  avg_views_per_viewer   numeric,
  new_viewers            bigint,
  subscribers_gained     int,
  subscribers_lost       int,
  likes                  int,
  dislikes               int,
  shares                 int,
  comments_added         int,
  views                  bigint,
  watch_time_hours       numeric,
  avg_view_duration_sec  int,
  unique_viewers         bigint,
  created_at             timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_yt_fact_video ON dw.fact_youtube_video_summary(video_sk);

-- 2) Views
CREATE OR REPLACE VIEW dw.vw_yt_kpis_total AS
SELECT
  COUNT(*)                      AS videos,
  SUM(f.impressions)            AS impressions,
  SUM(f.views)                  AS views,
  SUM(f.watch_time_hours)       AS watch_time_hours,
  AVG(f.impressions_ctr_pct)    AS avg_ctr_pct,
  AVG(f.avg_views_per_viewer)   AS avg_views_per_viewer
FROM dw.fact_youtube_video_summary f;

CREATE OR REPLACE VIEW dw.vw_yt_top_videos AS
SELECT
  dv.youtube_video_id,
  dv.title,
  dv.published_at,
  dv.category,
  f.views,
  f.impressions,
  f.impressions_ctr_pct,
  f.likes, f.dislikes, f.shares, f.comments_added,
  f.watch_time_hours,
  f.avg_view_duration_sec
FROM dw.fact_youtube_video_summary f
JOIN dw.dim_video dv ON dv.video_sk = f.video_sk
ORDER BY f.views DESC NULLS LAST;

CREATE OR REPLACE VIEW dw.vw_yt_by_publish_month AS
SELECT
  date_trunc('month', dv.published_at)::date AS month,
  COUNT(*) AS videos,
  SUM(f.views) AS views,
  SUM(f.impressions) AS impressions
FROM dw.fact_youtube_video_summary f
JOIN dw.dim_video dv ON dv.video_sk=f.video_sk
GROUP BY 1
ORDER BY 1;

-- 3) Sync routine (landing -> DW), NaN-safe + category fallback
CREATE OR REPLACE FUNCTION dw.sync_youtube_from_landing()
RETURNS void
LANGUAGE plpgsql
AS $dw$
BEGIN
  -- Upsert dim_video (dedup on platform+video_id)
  WITH src AS (
    SELECT
      (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='youtube') AS platform_sk,
      y.video_id::text        AS youtube_video_id,
      y.video_title           AS title,
      CASE
        WHEN y.publish_year IS NOT NULL AND y.publish_month IS NOT NULL AND y.publish_day IS NOT NULL
        THEN make_date(y.publish_year, y.publish_month, y.publish_day)
        ELSE NULL
      END                     AS published_date,
      y.duration::text        AS duration_text,
      COALESCE(NULLIF(TRIM(y.category::text),''), NULLIF(TRIM(y.content::text),'')) AS category,
      y.views, y.impressions
    FROM public.yt_video_etl y
    WHERE y.video_id IS NOT NULL AND y.video_id <> ''
  ),
  norm AS (
    SELECT
      platform_sk, youtube_video_id, title, published_date, category, views, impressions,
      CASE
        WHEN duration_text ~ '^\d+$'                 THEN duration_text::int
        WHEN duration_text ~ '^\d{1,2}:\d{2}:\d{2}$' THEN EXTRACT(EPOCH FROM duration_text::interval)::int
        WHEN duration_text ~ '^\d{1,2}:\d{2}$'       THEN EXTRACT(EPOCH FROM ('00:'||duration_text)::interval)::int
        WHEN duration_text ~ '^PT' THEN
             COALESCE((regexp_match(duration_text,'PT(\d+)H'))[1]::int,0)*3600
           + COALESCE((regexp_match(duration_text,'PT(?:\d+H)?(\d+)M'))[1]::int,0)*60
           + COALESCE((regexp_match(duration_text,'PT(?:\d+H)?(?:\d+M)?(\d+)S'))[1]::int,0)
        ELSE NULL
      END AS duration_sec
    FROM src
  ),
  dd AS (
    SELECT * FROM (
      SELECT n.*,
             ROW_NUMBER() OVER (
               PARTITION BY platform_sk, youtube_video_id
               ORDER BY COALESCE(n.views,0) DESC, COALESCE(n.impressions,0) DESC,
                        n.published_date DESC NULLS LAST
             ) rn
      FROM norm n
    ) z WHERE rn=1
  )
  INSERT INTO dw.dim_video(platform_sk, youtube_video_id, title, published_at, duration_sec, url, category)
  SELECT platform_sk,
         youtube_video_id,
         title,
         published_date,
         duration_sec,
         CASE WHEN youtube_video_id ~ '^[A-Za-z0-9_-]{11}$' THEN 'https://youtu.be/'||youtube_video_id END,
         category
  FROM dd
  ON CONFLICT (platform_sk, youtube_video_id) DO UPDATE
  SET title        = COALESCE(EXCLUDED.title,        dw.dim_video.title),
      published_at = COALESCE(EXCLUDED.published_at, dw.dim_video.published_at),
      duration_sec = COALESCE(EXCLUDED.duration_sec, dw.dim_video.duration_sec),
      url          = COALESCE(EXCLUDED.url,          dw.dim_video.url),
      category     = COALESCE(EXCLUDED.category,     dw.dim_video.category);

  -- Upsert fact (dedup + NaN-safe)
  WITH s_base AS (
    SELECT
      y.video_id::text AS youtube_video_id,
      ROW_NUMBER() OVER (
        PARTITION BY y.video_id
        ORDER BY COALESCE(y.views,0) DESC, COALESCE(y.impressions,0) DESC
      ) rn,
      y.impressions, y.impressions_ctr, y.avg_views_per_viewer,
      y.new_viewers, y.subscribers_gained, y.subscribers_lost,
      y.likes, y.dislikes, y.shares, y.comments_added, y.views,
      y.watch_time_hours,
      y.unique_viewers,
      (COALESCE(y.avg_dur_hours,0)*3600
       + COALESCE(y.avg_dur_minutes,0)*60
       + COALESCE(y.avg_dur_seconds,0))::int AS avg_view_duration_sec
    FROM public.yt_video_etl y
    WHERE y.video_id IS NOT NULL
  ),
  s_one AS (SELECT * FROM s_base WHERE rn=1),
  s_cast AS (
    SELECT
      youtube_video_id,
      CASE WHEN impressions::text            ~* '^(nan|inf|-inf)$' THEN NULL ELSE impressions END            AS impressions_n,
      CASE WHEN impressions_ctr::text        ~* '^(nan|inf|-inf)$' THEN NULL ELSE impressions_ctr END        AS impressions_ctr_n, -- 0..1
      CASE WHEN avg_views_per_viewer::text   ~* '^(nan|inf|-inf)$' THEN NULL ELSE avg_views_per_viewer END   AS avg_vpv_n,
      CASE WHEN watch_time_hours::text       ~* '^(nan|inf|-inf)$' THEN NULL ELSE watch_time_hours END       AS wth_n,
      CASE WHEN likes::text                  ~* '^(nan|inf|-inf)$' THEN NULL ELSE likes END                  AS likes_n,
      CASE WHEN dislikes::text               ~* '^(nan|inf|-inf)$' THEN NULL ELSE dislikes END               AS dislikes_n,
      CASE WHEN shares::text                 ~* '^(nan|inf|-inf)$' THEN NULL ELSE shares END                 AS shares_n,
      CASE WHEN comments_added::text         ~* '^(nan|inf|-inf)$' THEN NULL ELSE comments_added END         AS comments_n,
      CASE WHEN views::text                  ~* '^(nan|inf|-inf)$' THEN NULL ELSE views END                  AS views_n,
      unique_viewers,
      avg_view_duration_sec
    FROM s_one
  )
  INSERT INTO dw.fact_youtube_video_summary(
    video_sk,
    impressions, impressions_ctr_pct, avg_views_per_viewer, new_viewers,
    subscribers_gained, subscribers_lost, likes, dislikes, shares, comments_added,
    views, watch_time_hours, avg_view_duration_sec, unique_viewers
  )
  SELECT
    dv.video_sk,
    COALESCE(c.impressions_n,0)::bigint,
    (COALESCE(c.impressions_ctr_n,0)*100.0)::numeric,
    COALESCE(c.avg_vpv_n,0)::numeric,
    COALESCE(c.new_viewers,0)::bigint,
    COALESCE(c.subscribers_gained,0)::int,
    COALESCE(c.subscribers_lost,0)::int,
    COALESCE(c.likes_n,0)::int,
    COALESCE(c.dislikes_n,0)::int,
    COALESCE(c.shares_n,0)::int,
    COALESCE(c.comments_n,0)::int,
    COALESCE(c.views_n,0)::bigint,
    COALESCE(c.wth_n,0)::numeric,
    GREATEST(0, COALESCE(c.avg_view_duration_sec,0))::int,
    COALESCE(c.unique_viewers,0)::bigint
  FROM s_cast c
  JOIN dw.dim_video dv
    ON dv.platform_sk = (SELECT platform_sk FROM dw.dim_platform WHERE platform_code='youtube')
   AND dv.youtube_video_id = c.youtube_video_id
  ON CONFLICT (video_sk) DO UPDATE
  SET impressions           = EXCLUDED.impressions,
      impressions_ctr_pct   = EXCLUDED.impressions_ctr_pct,
      avg_views_per_viewer  = EXCLUDED.avg_views_per_viewer,
      new_viewers           = EXCLUDED.new_viewers,
      subscribers_gained    = EXCLUDED.subscribers_gained,
      subscribers_lost      = EXCLUDED.subscribers_lost,
      likes                 = EXCLUDED.likes,
      dislikes              = EXCLUDED.dislikes,
      shares                = EXCLUDED.shares,
      comments_added        = EXCLUDED.comments_added,
      views                 = EXCLUDED.views,
      watch_time_hours      = EXCLUDED.watch_time_hours,
      avg_view_duration_sec = EXCLUDED.avg_view_duration_sec,
      unique_viewers        = EXCLUDED.unique_viewers;
END;
$dw$;

-- 4) Trigger function + trigger (fires once per INSERT/UPDATE batch)
CREATE OR REPLACE FUNCTION dw.trg_yt_sync_stmt()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = dw, public
AS $$
BEGIN
  PERFORM dw.sync_youtube_from_landing();
  RETURN NULL;
END
$$;

DROP TRIGGER IF EXISTS yt_sync_stmt_insupd ON public.yt_video_etl;
CREATE TRIGGER yt_sync_stmt_insupd
AFTER INSERT OR UPDATE ON public.yt_video_etl
FOR EACH STATEMENT
EXECUTE FUNCTION dw.trg_yt_sync_stmt();
