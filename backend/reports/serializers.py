from rest_framework import serializers
from .models import Report, ReportTemplate

class ReportSerializer(serializers.ModelSerializer):
    report_type_display = serializers.CharField(source='get_report_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = Report
        fields = ['id', 'title', 'report_type', 'report_type_display', 'status', 
                 'status_display', 'date_range_start', 'date_range_end', 
                 'platforms', 'metrics', 'created_at', 'completed_at']
        read_only_fields = ['user', 'status', 'created_at', 'completed_at']

class ReportTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReportTemplate
        fields = '__all__'
