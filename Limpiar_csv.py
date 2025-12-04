import pandas as pd
import unicodedata


# ===  Eliminar columnas vacías ===  BUSCAR COMO FUNCIONA
def elimina_col_vacias(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def renombrar_columnas(df):
    df = df.rename(
        columns={
            "FECHA INGRESO": "FECHA_INGRESO",
            "TIPO CONSULTA": "TIPO_CONSULTA",
            "N° CASOS": "NUM_CASOS",
            "INSTITUCIÓN": "INSTITUCION",
            "MATERIA": "MATERIA",
            "SUBMATERIA": "SUBMATERIA",
        }
    )
    return df


def fechas_datetime(df):
    if "FECHA_INGRESO" in df.columns:
        df["FECHA_INGRESO"] = (
            df["FECHA_INGRESO"]
            .astype(str)
            .str.strip()
            .str.replace(r"[^0-9/\-]", "", regex=True)
        )
        df["FECHA_INGRESO"] = pd.to_datetime(
            df["FECHA_INGRESO"], errors="coerce", dayfirst=True, format="%d-%m-%Y"
        )
    return df


def quitar_espacios_iniciales(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def quitar_casos_duplicados(df):
    if "NUM_CASOS" in df.columns:
        df = df.drop_duplicates(subset=["NUM_CASOS"])
    return df


# =============== Elimianr tildes ===================
def quitar_tildes_df(df):
    def _strip_accents(texto):
        if isinstance(texto, str):
            texto = unicodedata.normalize("NFKD", texto)
            texto = texto.encode("ascii", "ignore").decode("utf-8")
        return texto

    df.columns = [_strip_accents(c) for c in df.columns]
    df = df.map(_strip_accents)
    return df


# =============== Comprobar tipos ===================
def trasformar_a_string(df):
    df["SUCURSAL"] = df["SUCURSAL"].astype(str)
    df["TIPO_CONSULTA"] = df["TIPO_CONSULTA"].astype(str)
    df["TIPO_CONSULTA"] = df["TIPO_CONSULTA"].astype(str)
    df["NUM_CASOS"] = df["NUM_CASOS"].astype(str)
    df["INSTITUCION"] = df["INSTITUCION"].astype(str)
    df["MATERIA"] = df["MATERIA"].astype(str)
    df["SUBMATERIA"] = df["SUBMATERIA"].astype(str)

    return df


def validar_fecha(df):
    if "FECHA_INGRESO" in df.columns:
        df = df[df["FECHA_INGRESO"].between("2025-01-01", "2025-12-31")]
    return df


def eliminar_nulos(df):
    columnas_claves = [
        "SUCURSAL",
        "TIPO_CONSULTA",
        "NUM_CASOS",
        "INSTITUCION",
        "MATERIA",
        "SUBMATERIA",
    ]
    df = df.dropna(subset=columnas_claves)
    return df


def funcion_ejecutor():
    df = pd.read_csv(
        "bd_presencial_agosto_2025.csv", sep=";", encoding="utf-8", dtype=str
    )
    df = elimina_col_vacias(df)
    df = renombrar_columnas(df)
    df = trasformar_a_string(df)
    df = quitar_espacios_iniciales(df)
    df = fechas_datetime(df)
    df = quitar_casos_duplicados(df)
    df = quitar_casos_duplicados(df)
    df = quitar_tildes_df(df)
    df = validar_fecha(df)
    df = eliminar_nulos(df)
    df.to_csv("nuevoCSV.csv", sep=";", index=False)
