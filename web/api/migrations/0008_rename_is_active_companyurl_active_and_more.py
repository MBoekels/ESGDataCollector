# Generated by Django 5.2.1 on 2025-06-05 11:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_companyprofile_active'),
    ]

    operations = [
        migrations.RenameField(
            model_name='companyurl',
            old_name='is_active',
            new_name='active',
        ),
        migrations.RenameField(
            model_name='pdffile',
            old_name='use_for_analysis',
            new_name='active',
        ),
    ]
