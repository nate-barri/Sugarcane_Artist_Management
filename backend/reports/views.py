from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.utils import timezone
from django.http import HttpResponse
from .models import Report, ReportTemplate
from .serializers import ReportSerializer, ReportTemplateSerializer
import json
from datetime import datetime

class ReportListCreateView(generics.ListCreateAPIView):
    serializer_class = ReportSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Report.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ReportDetailView(generics.RetrieveAPIView):
    serializer_class = ReportSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Report.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_report(request):
    """Generate a new report"""
    title = request.data.get('title', 'Dashboard Report')
    report_type = request.data.get('report_type', 'dashboard')
    
    # Create report instance
    report = Report.objects.create(
        user=request.user,
        title=title,
        report_type=report_type,
        date_range_start=timezone.now().date(),
        date_range_end=timezone.now().date(),
        platforms=['youtube', 'spotify', 'instagram'],
        metrics=['followers', 'views', 'engagement'],
        status='generating'
    )
    
    # In a real implementation, this would be handled by a background task
    # For now, we'll generate a simple report immediately
    report_data = {
        'summary': {
            'total_followers': 125000,
            'total_views': 2500000,
            'engagement_rate': 8.5,
            'growth_rate': 12.3
        },
        'platform_breakdown': {
            'youtube': {'followers': 75000, 'views': 1500000},
            'spotify': {'followers': 30000, 'streams': 800000},
            'instagram': {'followers': 20000, 'likes': 45000}
        },
        'generated_at': timezone.now().isoformat()
    }
    
    report.report_data = report_data
    report.status = 'completed'
    report.completed_at = timezone.now()
    report.save()
    
    serializer = ReportSerializer(report)
    return Response({
        'message': 'Report generated successfully',
        'report': serializer.data
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_report(request, report_id):
    """Download report as PDF"""
    try:
        report = Report.objects.get(id=report_id, user=request.user)
        
        if report.status != 'completed':
            return Response({'error': 'Report not ready'}, status=status.HTTP_400_BAD_REQUEST)
        
        # In a real implementation, this would generate and return a PDF
        # For now, return JSON data
        response = HttpResponse(
            json.dumps(report.report_data, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = f'attachment; filename="{report.title}.json"'
        return response
        
    except Report.DoesNotExist:
        return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)

class ReportTemplateListView(generics.ListAPIView):
    queryset = ReportTemplate.objects.filter(is_active=True)
    serializer_class = ReportTemplateSerializer
    permission_classes = [IsAuthenticated]
