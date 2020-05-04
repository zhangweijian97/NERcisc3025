from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from .algorithm.MEM import MEM

classifier = MEM()
classifier.load_model()


# Create your views here.
def index(request):
    context = {}

    query = request.GET.get('q', '')
    display_query = "INPUT: %s" % query
    context['display_query'] = display_query

    if query:
        predicted_words = classifier.predict_person(query)
        display_predicted_words = 'PERSON: %s' % predicted_words
        context['display_predicted_words'] = display_predicted_words
    # else:
    #     results = []

    return render(request, 'NER/index.html', context)
