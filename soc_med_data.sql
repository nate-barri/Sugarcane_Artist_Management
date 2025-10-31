select*from public.social_media_data

truncate table public.social_media_data 
drop table if exists public.social_media_data

SELECT column_name FROM information_schema.columns WHERE table_name = 'social_media_data';
