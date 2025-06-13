from django import forms
from django.forms import inlineformset_factory
from .models import CompanyProfile, CompanyURL, Query

class CompanyProfileForm(forms.ModelForm):
    class Meta:
        model = CompanyProfile
        fields = ['name', 'industry', 'info']

class CompanyURLForm(forms.ModelForm):
    class Meta:
        model = CompanyURL
        fields = ["url", "active"]
        widgets = {
            'url': forms.URLInput(attrs={'class': 'form-control'}),
            'active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class QueryForm(forms.ModelForm):
    class Meta:
        model = Query
        fields = ['name', 'description', 'question', 'active']

CompanyURLFormSet = inlineformset_factory(
    CompanyProfile,
    CompanyURL,
    form=CompanyURLForm,
    fields=('url', 'active'),
    extra=0,
    can_delete=True
)

