# Generated by Django 5.2.1 on 2025-06-04 15:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_query_last_edited'),
    ]

    operations = [
        migrations.AddField(
            model_name='companyprofile',
            name='active',
            field=models.BooleanField(default=True),
        ),
    ]
