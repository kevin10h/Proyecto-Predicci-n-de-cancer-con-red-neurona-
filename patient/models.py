from django.db import models

# Create your models here.


class Persona(models.Model):
    SEXO_CHOICES = [
        ('F', 'Femenino'),
        ('M', 'Masculino'),
    ]

    ESTADO_CIVIL_CHOICES = [
        ('Soltero', 'Soltero'),
        ('Casado', 'Casado'),
        ('Viudo', 'Viudo'),
        ('Divorciado', 'Divorciado'),
    ]

    ORIENTACION_SEXUAL_CHOICES = [
        ('Heterosexual', 'Heterosexual'),
        ('Bisexual', 'Bisexual'),
        ('Homosexual', 'Homosexual'),
    ]

    NIVEL_SOCIOECONOMICO_CHOICES = [
        ('Bajo', 'Bajo'),
        ('Medio', 'Medio'),
        ('Alto', 'Alto'),
    ]

    edad = models.PositiveIntegerField()
    sexo = models.CharField(max_length=1, choices=SEXO_CHOICES)
    estado_civil = models.CharField(max_length=10, choices=ESTADO_CIVIL_CHOICES)
    orientacion_sexual = models.CharField(max_length=12, choices=ORIENTACION_SEXUAL_CHOICES)
    uso_proteccion = models.BooleanField()
    consumo_drogas = models.BooleanField()
    n_parejas_sexuales = models.PositiveIntegerField()
    historial_its = models.BooleanField()
    nivel_socioeconomico = models.CharField(max_length=10, choices=NIVEL_SOCIOECONOMICO_CHOICES)
    cantidad_tatuajes = models.PositiveIntegerField()
    cantidad_donaciones_dadas = models.PositiveIntegerField()
    cantidad_donaciones_recibidas = models.PositiveIntegerField()

    resultado_prediccion = models.BooleanField(null=True, blank=True)
    porcentaje_confianza = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.sexo} - {self.edad} años"