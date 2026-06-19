from django.contrib import admin
from .models import SearchHistory, SummaryHistory

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display  = ('user_email', 'query', 'results_count', 'had_error', 'searched_at')
    list_filter   = ('had_error',)
    search_fields = ('user_email', 'query')

@admin.register(SummaryHistory)
class SummaryHistoryAdmin(admin.ModelAdmin):
    list_display  = ('user_email', 'filename', 'summary_word_count', 'summarized_at')
    search_fields = ('user_email', 'filename')