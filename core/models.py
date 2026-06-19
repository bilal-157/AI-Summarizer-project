from django.db import models
from django.contrib.auth.models import User


class SearchHistory(models.Model):
    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name='searches')
    query         = models.CharField(max_length=255)
    results_count = models.IntegerField(default=0)
    searched_at   = models.DateTimeField(auto_now_add=True)
    user_email    = models.EmailField()
    had_error     = models.BooleanField(default=False)

    class Meta:
        ordering = ['-searched_at']

    def __str__(self):
        return f"{self.user_email} | {self.query} ({self.searched_at:%Y-%m-%d})"


class SummaryHistory(models.Model):
    user               = models.ForeignKey(User, on_delete=models.CASCADE, related_name='summaries')
    filename           = models.CharField(max_length=255)
    summary_word_count = models.IntegerField(default=0)
    summarized_at      = models.DateTimeField(auto_now_add=True)
    user_email         = models.EmailField()

    class Meta:
        ordering = ['-summarized_at']

    def __str__(self):
        return f"{self.user_email} | {self.filename} ({self.summarized_at:%Y-%m-%d})"