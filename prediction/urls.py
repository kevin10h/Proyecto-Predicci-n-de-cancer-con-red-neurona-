from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView


urlpatterns = [
    path('admin/', admin.site.urls),
	path('patient/', include('patient.urls')),
    path('', RedirectView.as_view(url='patient/', permanent=False)),  # 👈 Esta línea redirige la raíz a /patient/
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# serve static files
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)