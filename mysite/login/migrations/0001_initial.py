# Generated by Django 5.0.1 on 2024-01-10 11:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('username', models.CharField(max_length=6, primary_key=True, serialize=False)),
                ('password', models.IntegerField(max_length=4)),
                ('mail', models.EmailField(max_length=30)),
            ],
        ),
    ]